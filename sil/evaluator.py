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

"""Custom evaluator for soft imitation learning setting.

Includes the estimated return from the learned reward function.

Also captures videos.
"""
from typing import Dict, Optional, Sequence, Union

from absl import logging
import acme
from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax import utils
from acme.jax import variable_utils
from acme.jax.experiments import config as experiment_config
from acme.utils import counting
from acme.utils import experiment_utils
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.wrappers import video as video_wrappers
import dm_env
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import tree

from sil import config as sil_config
from sil import networks as sil_networks


N_EVALS_PER_VIDEO = 1


class GymVideoWrapper(video_wrappers.VideoWrapper):

  def _render_frame(self, observation):
    """Renders a frame from the given environment observation."""
    del observation
    return self.environment.render(mode='rgb_array')  # pytype: disable=attribute-error

  def _write_frames(self):
    logging.info(
        'Saving video: %s/%s/%d', self._path, self._filename, self._counter
    )
    super()._write_frames()


class RewardCore:
  """A learned implicit or explicit reward function."""

  def __init__(
      self,
      networks: sil_networks.SILNetworks,
      reward_factory: sil_config.RewardFact,
      discount: float,
      variable_source: core.VariableSource,
  ):
    self._reward_variable_client = variable_utils.VariableClient(
        variable_source, 'reward', device='cpu'
    )
    self._critic_variable_client = variable_utils.VariableClient(
        variable_source, 'critic', device='cpu'
    )
    self._policy_variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device='cpu'
    )
    self._reward_fn = None

    def _reward(
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
        transition_discount: jnp.ndarray,
        key: jnp.ndarray,
        reward_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        policy_params: networks_lib.Params,
    ) -> jnp.ndarray:
      def state_action_reward_fn(
          state: jnp.ndarray, action: jnp.ndarray
      ) -> jnp.ndarray:
        return jnp.ravel(
            networks.reward_network.apply(reward_params, state, action)
        )

      def state_action_value_fn(
          state: jnp.ndarray, action: jnp.ndarray
      ) -> jnp.ndarray:
        return networks.critic_network.apply(
            critic_params, state, action).min(axis=-1)

      def state_value_fn(
          state: jnp.ndarray, policy_key: jnp.ndarray
      ) -> jnp.ndarray:
        action_dist = networks.policy_network.apply(policy_params, state)
        action = action_dist.sample(seed=policy_key)
        v = networks.critic_network.apply(
            critic_params, state, action).min(axis=-1)
        return v

      reward = reward_factory(
          state_action_reward_fn,
          state_action_value_fn,
          state_value_fn,
          discount,
      )
      return reward(state, action, next_state, transition_discount, key)

    self._reward_fn = jax.jit(_reward)

  @property
  def _reward_params(self) -> Sequence[networks_lib.Params]:
    return self._reward_variable_client.params

  @property
  def _critic_params(self) -> Sequence[networks_lib.Params]:
    return self._critic_variable_client.params

  @property
  def _policy_params(self) -> Sequence[networks_lib.Params]:
    params = self._policy_variable_client.params
    return params

  def __call__(
      self,
      state: jnp.ndarray,
      action: jnp.ndarray,
      next_state: jnp.ndarray,
      discount: jnp.ndarray,
      key: jnp.ndarray,
  ) -> jnp.ndarray:
    assert self._reward_fn is not None
    return self._reward_fn(
        state,
        action,
        next_state,
        discount,
        key,
        self._reward_params,
        self._critic_params,
        self._policy_params,
    )

  def update(self, wait: bool = False):
    """Get updated parameters and update reward function."""

    self._reward_variable_client.update(wait)
    self._critic_variable_client.update(wait)
    self._policy_variable_client.update(wait)


class ImitationObserver(observers_lib.EnvLoopObserver):
  """Observer that evaluated using the learned reward function."""

  def __init__(self, reward_fn: RewardCore):
    self._reward_fn = reward_fn
    self._imitation_return = 0.0
    self._current_observation = None
    self._episode_rewards = []
    self._epsiode_imitation_rewards = []
    self._key = random.PRNGKey(0)

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep):
    self._reward_fn.update()
    self._imitation_return = 0.0
    self._current_observation = timestep.observation
    self._episode_rewards = []
    self._epsiode_imitation_rewards = []

  def observe(
      self,
      env: dm_env.Environment,
      timestep: dm_env.TimeStep,
      action: np.ndarray,
  ):
    """Records one environment step."""
    self._key, subkey = random.split(self._key)
    obs = tree.map_structure(
        lambda obs: jnp.expand_dims(obs, 0), self._current_observation
    )
    next_obs = tree.map_structure(
        lambda obs: jnp.expand_dims(obs, 0), timestep.observation
    )
    imitation_reward = self._reward_fn(
        obs,
        jnp.expand_dims(action, 0),
        next_obs,
        jnp.expand_dims(timestep.discount, 0),
        subkey,
    ).squeeze()
    self._current_observation = timestep.observation
    self._imitation_return += imitation_reward
    self._episode_rewards += [timestep.reward]
    self._epsiode_imitation_rewards += [imitation_reward]

  def compute_correlation(self) -> jnp.ndarray:
    r = jnp.asarray(self._episode_rewards)
    ir = jnp.asarray(self._epsiode_imitation_rewards)
    return jnp.corrcoef(r, ir)[0, 1]

  def get_metrics(self) -> Dict[str, observers_lib.Number]:
    """Returns metrics collected for the current episode."""
    corr = self.compute_correlation()
    metrics = {
        'imitation_return': self._imitation_return,
        'episode_reward_corr': corr,
    }
    return metrics  # pytype: disable=bad-return-type  # jnp-array


def adroit_success(env: dm_env.Environment) -> bool:
  # Adroit environments have get_info methods.
  if not hasattr(env, 'get_info'):
    return False
  info = getattr(env, 'get_info')()
  if 'goal_achieved' in info:
    return info['goal_achieved']
  else:
    return False


def _get_success_from_env(env: dm_env.Environment) -> bool:
  """Obtain the success flag for Adroit environments."""
  return adroit_success(env)


class SuccessObserver(observers_lib.EnvLoopObserver):
  """Observer that extracts the goal_achieved flag from Adroit mj_env tasks."""

  _success: bool = False

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep):
    del env, timestep
    self._success = False

  def observe(
      self,
      env: dm_env.Environment,
      timestep: dm_env.TimeStep,
      action: np.ndarray,
  ):
    """Records one environment step."""
    del timestep, action
    success = _get_success_from_env(env)
    self._success = self._success or success

  def get_metrics(self) -> Dict[str, observers_lib.Number]:
    """Returns metrics collected for the current episode."""
    metrics = {
        'success': 1.0 if self._success else 0.0,
    }
    return metrics


def imitation_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: experiment_config.NetworkFactory[builders.Networks],
    policy_factory: experiment_config.PolicyFactory[
        builders.Networks, builders.Policy
    ],
    logger_factory: loggers.LoggerFactory = experiment_utils.make_experiment_logger,
    agent_config: Union[sil_config.SILConfig, None] = None,
) -> experiment_config.EvaluatorFactory[builders.Policy]:
  """Returns an imitation learning evaluator process."""

  def evaluator(
      random_key: types.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: experiment_config.MakeActorFn[builders.Policy],
  ) -> environment_loop.EnvironmentLoop:
    """The evaluation process for imitation learning."""
    # Create environment and evaluator networks.
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))

    eval_environment = environment

    environment_spec = specs.make_environment_spec(environment)
    networks = network_factory(environment_spec)
    policy = policy_factory(networks, environment_spec, True)
    actor = make_actor(actor_key, policy, environment_spec, variable_source)

    success_observer = SuccessObserver()
    if agent_config is not None:
      reward_factory = agent_config.imitation.reward_factory()
      reward_fn = RewardCore(
          networks, reward_factory, agent_config.discount, variable_source
      )

      imitation_observer = ImitationObserver(reward_fn)
      observers = (success_observer, imitation_observer)
    else:
      observers = (success_observer,)

    # Create logger and counter.
    counter = counting.Counter(counter, 'imitation_evaluator')
    logger = logger_factory('imitation_evaluator', 'actor_steps', 0)

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        eval_environment, actor, counter, logger, observers=observers
    )

  return evaluator


def video_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: experiment_config.NetworkFactory[builders.Networks],
    policy_factory: experiment_config.PolicyFactory[
        builders.Networks, builders.Policy
    ],
    logger_factory: loggers.LoggerFactory = experiment_utils.make_experiment_logger,
    videos_per_eval: int = 0,
) -> experiment_config.EvaluatorFactory[builders.Policy]:
  """Returns an evaluator process that records videos."""

  def evaluator(
      random_key: types.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: experiment_config.MakeActorFn[builders.Policy],
  ) -> environment_loop.EnvironmentLoop:
    """The evaluation process for recording videos."""
    # Create environment and evaluator networks.
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))

    if videos_per_eval > 0:
      eval_environment = GymVideoWrapper(
          environment,
          record_every=videos_per_eval,
          frame_rate=40,
          filename='eval_episode',
      )
    else:
      eval_environment = environment

    environment_spec = specs.make_environment_spec(environment)
    networks = network_factory(environment_spec)
    policy = policy_factory(networks, environment_spec, True)
    actor = make_actor(actor_key, policy, environment_spec, variable_source)

    observers = (SuccessObserver(),)

    # Create logger and counter.
    counter = counting.Counter(counter, 'video_evaluator')
    logger = logger_factory(
        'video_evaluator', 'actor_steps', 0
    )

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        eval_environment, actor, counter, logger, observers=observers
    )

  return evaluator


def bc_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: experiment_config.NetworkFactory[builders.Networks],
    policy_factory: experiment_config.PolicyFactory[
        builders.Networks, builders.Policy
    ],
    logger_factory: loggers.LoggerFactory = experiment_utils.make_experiment_logger,
) -> experiment_config.EvaluatorFactory[builders.Policy]:
  """Returns an imitation learning evaluator process."""

  def evaluator(
      random_key: types.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: experiment_config.MakeActorFn[builders.Policy],
  ) -> environment_loop.EnvironmentLoop:
    del make_actor
    # Create environment and evaluator networks.
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))

    eval_environment = environment

    environment_spec = specs.make_environment_spec(environment)
    networks = network_factory(environment_spec)
    policy = policy_factory(networks, environment_spec, True)

    def make_bc_actor(
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
    ) -> acme.Actor:
      del environment_spec
      assert variable_source is not None
      actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
      variable_client = variable_utils.VariableClient(
          variable_source, 'bc_policy_params', device='cpu'
      )
      return actors.GenericActor(
          actor_core, random_key, variable_client, None, backend='cpu'
      )

    actor = make_bc_actor(actor_key, policy, environment_spec, variable_source)

    observers = (SuccessObserver(),)

    # Create logger and counter.
    counter = counting.Counter(counter, 'bc_evaluator')
    logger = logger_factory('bc_evaluator', 'actor_steps', 0)

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        eval_environment, actor, counter, logger, observers=observers
    )

  return evaluator
