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

"""Example running coherent soft imitation learning on continuous control tasks."""

import math
from typing import Iterator

from absl import flags
from acme import specs
from acme import types
from acme.agents.jax import sac
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import jax.random as rand
import launchpad as lp

from csil.scripts import helpers
from csil.sil import builder
from csil.sil import config as sil_config
from csil.sil import evaluator
from csil.sil import networks

USE_SARSA = True

_DIST_FLAG = flags.DEFINE_bool(
    'run_distributed',
    False,
    (
        'Should an agent be executed in a distributed '
        'way. If False, will run single-threaded.'
    ),
)
_ENV_NAME = flags.DEFINE_string(
    'env_name', 'HalfCheetah-v2', 'Which environment to run'
)
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_N_STEPS = flags.DEFINE_integer(
    'num_steps', 250_000, 'Number of env steps to run.'
)
_EVAL_RATIO = flags.DEFINE_integer(
    'eval_every', 1_000, 'How often to evaluate for local runs.'
)
_N_EVAL_EPS = flags.DEFINE_integer(
    'evaluation_episodes', 1, 'Evaluation episodes for local runs.'
)
_N_DEMONSTRATIONS = flags.DEFINE_integer(
    'num_demonstrations', 25, 'No. of demonstration trajectories.'
)
_N_OFFLINE_DATASET = flags.DEFINE_integer(
    'num_offline_demonstrations', 1_000, 'Offline dataset size.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 256, 'Batch size.')
_ENT_COEF = flags.DEFINE_float('entropy_coefficient', 0.01, 'Temperature')
_ENT_SF = flags.DEFINE_float('ent_sf', 1.0, 'Scale entropy target.')
_DAMP = flags.DEFINE_float('damping', 0.0, 'Constraint damping.')
_SF = flags.DEFINE_float('scale_factor', 1.0, 'Reward loss scale factor.')
_GNSF = flags.DEFINE_float(
    'grad_norm_scale_factor', 1.0, 'Critic grad scale factor.'
)
_DISCOUNT = flags.DEFINE_float('discount', 0.99, 'Discount factor')
_ACTOR_LR = flags.DEFINE_float('actor_lr', 3e-4, 'Actor learning rate.')
_CRITIC_LR = flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate.')
_REWARD_LR = flags.DEFINE_float('reward_lr', 1e-3, 'Reward learning rate.')
_TAU = flags.DEFINE_float(
    'tau',
    0.005,
    (
        'Target network exponential smoothing weight.'
        '1. = no update, 0, = no smoothing.'
    ),
)
_CRITIC_ACTOR_RATIO = flags.DEFINE_integer(
    'critic_actor_update_ratio', 1, 'Critic updates per actor update.'
)
_SGD_STEPS = flags.DEFINE_integer(
    'sgd_steps', 1, 'SGD steps for online sample.'
)
_CRITIC_NETWORK = flags.DEFINE_multi_integer(
    'critic_network', [256, 256], 'Define critic architecture.'
)
_REWARD_NETWORK = flags.DEFINE_multi_integer(
    'reward_network', [256, 256], 'Define reward architecture. (Unused)'
)
_POLICY_NETWORK = flags.DEFINE_multi_integer(
    'policy_network', [256, 256, 12, 256], 'Define policy architecture.'
)
_POLICY_MODEL = flags.DEFINE_enum(
    'policy_model',
    networks.PolicyArchitectures.HETSTATTRI.value,
    [e.value for e in networks.PolicyArchitectures],
    'Define policy model type.',
)
_LAYERNORM = flags.DEFINE_bool(
    'policy_layer_norm', False, 'Use layer norm for first layer of the policy.'
)
_CRITIC_MODEL = flags.DEFINE_enum(
    'critic_model',
    networks.CriticArchitectures.LNMLP.value,
    [e.value for e in networks.CriticArchitectures],
    'Define critic model type.',
)
_REWARD_MODEL = flags.DEFINE_enum(
    'reward_model',
    networks.RewardArchitectures.PCSIL.value,
    [e.value for e in networks.RewardArchitectures],
    'Define reward model type.',
)
_RCALE = flags.DEFINE_float('reward_scaling', 1.0, 'Scale learned reward.')
_FINETUNE_R = flags.DEFINE_bool('finetune_reward', True, 'Finetune reward.')
_LOSS_TYPE = flags.DEFINE_enum(
    'loss_type',
    sil_config.Losses.FAITHFUL.value,
    [e.value for e in sil_config.Losses],
    'Define regression loss type.',
)
_POLICY_PRETRAIN_STEPS = flags.DEFINE_integer(
    'policy_pretrain_steps', 25_000, 'Policy pretraining steps.'
)
_POLICY_PRETRAIN_LR = flags.DEFINE_float(
    'policy_pretrain_lr', 1e-3, 'Policy pretraining learning rate.'
)
_CRITIC_PRETRAIN_STEPS = flags.DEFINE_integer(
    'critic_pretrain_steps', 5_000, 'Critic pretraining steps.'
)
_CRITIC_PRETRAIN_LR = flags.DEFINE_float(
    'critic_pretrain_lr', 1e-4, 'Critic pretraining learning rate.'
)
_OFFLINE_FLAG = flags.DEFINE_bool('offline', False, 'Run an offline agent.')
_EVAL_PER_VIDEO = flags.DEFINE_integer(
    'evals_per_video', 0, 'Video frequency. Disable using 0.'
)
_NUM_ACTORS = flags.DEFINE_integer(
    'num_actors', 4, 'Number of distributed actors.'
)


def _build_experiment_config():
  """Builds a CSIL experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.

  task = _ENV_NAME.value

  make_env, env_spec, make_demonstrations = helpers.get_env_and_demonstrations(
      task, _N_DEMONSTRATIONS.value, use_sarsa=USE_SARSA,
      in_memory='image' not in task
  )

  def environment_factory(seed: int):
    del seed
    return make_env()

  batch_size = _BATCH_SIZE.value
  seed = _SEED.value
  actor_lr = _ACTOR_LR.value

  make_demonstrations_ = lambda batchsize: make_demonstrations(batchsize, seed)

  if _ENT_COEF.value > 0.0:
    kwargs = {'entropy_coefficient': _ENT_COEF.value}
  else:
    target_entropy = _ENT_SF.value * sac.target_entropy_from_env_spec(
        env_spec, target_entropy_per_dimension=abs(_ENT_SF.value))
    kwargs = {'target_entropy': target_entropy}

  # Important step that normalizes reward values -- do not change!
  csil_alpha = _RCALE.value / math.prod(env_spec.actions.shape)

  policy_architecture = networks.PolicyArchitectures(_POLICY_MODEL.value)
  bc_policy_architecture = policy_architecture
  critic_architecture = networks.CriticArchitectures(_CRITIC_MODEL.value)
  reward_architecture = networks.RewardArchitectures(_REWARD_MODEL.value)
  policy_layers = _POLICY_NETWORK.value
  reward_layers = _REWARD_NETWORK.value
  critic_layers = _CRITIC_NETWORK.value
  use_layer_norm = _LAYERNORM.value

  def network_factory(spec: specs.EnvironmentSpec):
    return networks.make_networks(
        spec=spec,
        reward_policy_coherence_alpha=csil_alpha,
        policy_architecture=policy_architecture,
        critic_architecture=critic_architecture,
        reward_architecture=reward_architecture,
        bc_policy_architecture=bc_policy_architecture,
        policy_hidden_layer_sizes=tuple(policy_layers),
        reward_hidden_layer_sizes=tuple(reward_layers),
        critic_hidden_layer_sizes=tuple(critic_layers),
        bc_policy_hidden_layer_sizes=tuple(policy_layers),
        layer_norm_policy=use_layer_norm,
    )

  demo_factory = lambda seed_: make_demonstrations(batch_size, seed_)
  policy_pretraining = sil_config.PretrainingConfig(
      loss=sil_config.Losses(_LOSS_TYPE.value),
      seed=seed,
      dataset_factory=demo_factory,
      steps=_POLICY_PRETRAIN_STEPS.value,
      learning_rate=_POLICY_PRETRAIN_LR.value,
      use_as_reference=True,
  )
  if _OFFLINE_FLAG.value:
    _, offline_dataset = helpers.get_offline_dataset(
        task,
        env_spec,
        _N_DEMONSTRATIONS.value,
        _N_OFFLINE_DATASET.value,
        use_sarsa=USE_SARSA,
    )

    def offline_pretraining_dataset(rseed: int) -> Iterator[types.Transition]:
      rkey = rand.PRNGKey(rseed)
      return helpers.MixedIterator(
          offline_dataset(batch_size, rkey),
          make_demonstrations(batch_size, rseed),
      )

    offline_policy_pretraining = sil_config.PretrainingConfig(
        loss=sil_config.Losses(_LOSS_TYPE.value),
        seed=seed,
        dataset_factory=offline_pretraining_dataset,
        steps=_POLICY_PRETRAIN_STEPS.value,
        learning_rate=_POLICY_PRETRAIN_LR.value,
    )
    policy_pretrainers = [offline_policy_pretraining, policy_pretraining]
    critic_dataset = demo_factory
  else:
    policy_pretrainers = [
        policy_pretraining,
    ]
    critic_dataset = demo_factory

  critic_pretraining = sil_config.PretrainingConfig(
      seed=seed,
      dataset_factory=critic_dataset,
      steps=_CRITIC_PRETRAIN_STEPS.value,
      learning_rate=_CRITIC_PRETRAIN_LR.value,
  )
  # Construct the agent.
  config_ = sil_config.SILConfig(
      imitation=sil_config.CoherentConfig(
          reward_scaling=_RCALE.value,
          refine_reward=_FINETUNE_R.value,
          negative_reward=(
              reward_architecture == networks.RewardArchitectures.NCSIL),
          grad_norm_sf=_GNSF.value,
          scale_factor=_SF.value,
      ),
      actor_bc_loss=False,
      policy_pretraining=policy_pretrainers,
      critic_pretraining=critic_pretraining,
      expert_demonstration_factory=make_demonstrations_,
      discount=_DISCOUNT.value,
      critic_learning_rate=_CRITIC_LR.value,
      reward_learning_rate=_REWARD_LR.value,
      actor_learning_rate=actor_lr,
      num_sgd_steps_per_step=_SGD_STEPS.value,
      critic_actor_update_ratio=_CRITIC_ACTOR_RATIO.value,
      n_step=1,
      damping=_DAMP.value,
      tau=_TAU.value,
      batch_size=batch_size,
      samples_per_insert=batch_size,
      alpha_learning_rate=1e-2,
      alpha_init=0.01,
      **kwargs,
  )

  sil_builder = builder.SILBuilder(config_)

  imitation_evaluator_factory = evaluator.imitation_evaluator_factory(
      agent_config=config_,
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_factory=sil_builder.make_policy,
  )

  bc_evaluator_factory = evaluator.bc_evaluator_factory(
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_factory=sil_builder.make_bc_policy,
  )

  if _EVAL_PER_VIDEO.value > 0:
    video_evaluator_factory = evaluator.video_evaluator_factory(
        environment_factory=environment_factory,
        network_factory=network_factory,
        policy_factory=sil_builder.make_policy,
        videos_per_eval=_EVAL_PER_VIDEO.value,
    )
    evaluators = [imitation_evaluator_factory, bc_evaluator_factory,
                  video_evaluator_factory]
  else:
    evaluators = [imitation_evaluator_factory, bc_evaluator_factory]

  if _OFFLINE_FLAG.value:
    make_offline_dataset, _ = helpers.get_offline_dataset(
        task,
        env_spec,
        _N_DEMONSTRATIONS.value,
        _N_OFFLINE_DATASET.value,
        use_sarsa=USE_SARSA,
    )
    # Only uses random key, to bake the batch size in.
    make_offline_dataset_ = lambda rk: make_offline_dataset(batch_size, rk)
    return experiments.OfflineExperimentConfig(
        builder=sil_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        demonstration_dataset_factory=make_offline_dataset_,
        evaluator_factories=evaluators,
        max_num_learner_steps=_N_STEPS.value,
        environment_spec=env_spec,
        seed=_SEED.value,
    )
  else:
    return experiments.ExperimentConfig(
        builder=sil_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        evaluator_factories=evaluators,
        seed=_SEED.value,
        max_num_actor_steps=_N_STEPS.value,
    )


def main(_):
  config = _build_experiment_config()
  if _DIST_FLAG.value:
    if _OFFLINE_FLAG.value:
      program = experiments.make_distributed_offline_experiment(
          experiment=config
      )
    else:
      program = experiments.make_distributed_experiment(
          experiment=config, num_actors=_NUM_ACTORS.value
      )
    lp.launch(
        program,
        xm_resources=lp_utils.make_xm_docker_resources(program),
    )
  else:
    if _OFFLINE_FLAG.value:
      experiments.run_offline_experiment(
          experiment=config,
          eval_every=_EVAL_RATIO.value,
          num_eval_episodes=_N_EVAL_EPS.value,
      )
    else:
      experiments.run_experiment(
          experiment=config,
          eval_every=_EVAL_RATIO.value,
          num_eval_episodes=_N_EVAL_EPS.value,
      )


if __name__ == '__main__':
  app.run(main)
