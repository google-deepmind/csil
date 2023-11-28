# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Soft imitation learning builder."""
from typing import Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.sac import networks as sac_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb
from reverb import rate_limiters

from sil import config as sil_config
from sil import learning
from sil import networks as sil_networks


class SILBuilder(
    builders.ActorLearnerBuilder[
        sil_networks.SILNetworks,
        actor_core_lib.FeedForwardPolicy,
        learning.ImitationSample,
    ]
):
  """Soft Imitation Learning Builder."""

  def __init__(self, config: sil_config.SILConfig):
    """Creates a soft imitation learner, a behavior policy and an eval actor.

    Args:
      config: a config with hyperparameters
    """
    self._config = config
    self._make_demonstrations = config.expert_demonstration_factory

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: sil_networks.SILNetworks,
      dataset: Iterator[learning.ImitationSample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    # Create optimizers.
    policy_optimizer = optax.adam(
        learning_rate=self._config.actor_learning_rate
    )
    q_optimizer = optax.adam(self._config.critic_learning_rate)
    r_optimizer = optax.sgd(learning_rate=self._config.reward_learning_rate)

    critic_loss = self._config.imitation.critic_loss_factory()
    reward_factory = self._config.imitation.reward_factory()

    n_policy_pretrainers = (len(self._config.policy_pretraining)
                            if self._config.policy_pretraining else 0)
    policy_pretraining_loggers = [
      logger_fn(f'pretrainer_policy{i}')
      for i in range(n_policy_pretrainers)
    ]

    return learning.SILLearner(
        networks=networks,
        critic_loss_def=critic_loss,
        reward_factory=reward_factory,
        tau=self._config.tau,
        discount=self._config.discount,
        critic_actor_update_ratio=self._config.critic_actor_update_ratio,
        entropy_coefficient=self._config.entropy_coefficient,
        target_entropy=self._config.target_entropy,
        alpha_init=self._config.alpha_init,
        alpha_learning_rate=self._config.alpha_learning_rate,
        damping=self._config.damping,
        rng=random_key,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        r_optimizer=r_optimizer,
        dataset=dataset,
        actor_bc_loss=self._config.actor_bc_loss,
        policy_pretraining=self._config.policy_pretraining,
        critic_pretraining=self._config.critic_pretraining,
        learner_logger=logger_fn('learner'),
        policy_pretraining_loggers=policy_pretraining_loggers,
        critic_pretraining_logger=logger_fn('pretrainer_critic'),
        counter=counter,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: actor_core_lib.FeedForwardPolicy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
    variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device='cpu')
    return actors.GenericActor(
        actor_core, random_key, variable_client, adder, backend='cpu')

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: actor_core_lib.FeedForwardPolicy,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    del policy
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate *
        self._config.samples_per_insert)
    error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=self._config.min_replay_size,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer)
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec))
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client
  ) -> Iterator[learning.ImitationSample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    # Replay buffer for demonstration data.
    iterator_demos = self._make_demonstrations(self._config.batch_size)

    # Replay buffer for online experience.
    iterator_online = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._config.batch_size,
        prefetch_size=self._config.prefetch_size,
    ).as_numpy_iterator()

    return utils.device_put(
        (
            learning.ImitationSample(types.Transition(*online.data), demo)
            for online, demo in zip(iterator_online, iterator_demos)
        ),
        jax.devices()[0],
    )

  def make_adder(
      self, replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[actor_core_lib.FeedForwardPolicy]
  ) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment."""
    del environment_spec, policy
    return adders_reverb.NStepTransitionAdder(
        priority_fns={self._config.replay_table_name: None},
        client=replay_client,
        n_step=self._config.n_step,
        discount=self._config.discount,
    )

  def make_policy(
      self,
      networks: sil_networks.SILNetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False,
  ) -> actor_core_lib.FeedForwardPolicy:
    """Construct the policy, which is the same as soft actor critic's policy."""
    del environment_spec
    return sac_networks.apply_policy_and_sample(
        networks.to_sac(using_bc_policy=False), eval_mode=evaluation
    )

  def make_bc_policy(
      self,
      networks: sil_networks.SILNetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False,
  ) -> actor_core_lib.FeedForwardPolicy:
    """Construct the policy, which is the same as soft actor critic's policy."""
    del environment_spec
    return sac_networks.apply_policy_and_sample(
        networks.to_sac(using_bc_policy=True), eval_mode=evaluation
    )
