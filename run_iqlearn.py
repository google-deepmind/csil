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

"""Example running IQ-Learn on continuous control tasks."""

from absl import flags
from acme import specs
from acme.agents.jax import sac
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import jax
import launchpad as lp

import experiment_logger
import helpers
from sil import builder
from sil import config as sil_config
from sil import evaluator
from sil import networks


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
    'eval_every', 5_000, 'How often to run evaluation.'
)
_N_EVAL_EPS = flags.DEFINE_integer(
    'evaluation_episodes', 1, 'Evaluation episodes.'
)
_N_DEMOS = flags.DEFINE_integer(
    'num_demonstrations', 25, 'Number of demonstration trajectories.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 256, 'Batch size.')
_N_OFFLINE_DATASET = flags.DEFINE_integer(
    'num_offline_demonstrations', 25, 'Offline dataset size.'
)
_DISCOUNT = flags.DEFINE_float('discount', 0.99, 'Discount factor')
_ACTOR_LR = flags.DEFINE_float('actor_lr', 3e-5, 'Actor learning rate.')
_CRITIC_LR = flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate.')
_REWARD_LR = flags.DEFINE_float(
    'reward_lr', 3e-4, 'Reward learning rate (unused).'
)
_ENT_COEF = flags.DEFINE_float(
    'entropy_coefficient',
    0.01,
    'Entropy coefficient. Becomes adaptive if None.',
)
_TAU = flags.DEFINE_float(
    'tau',
    0.005,
    (
        'Target network exponential smoothing weight.'
        '1. = no update, 0, = no smoothing.'
    ),
)
_OFFLINE_FLAG = flags.DEFINE_bool('offline', False, 'Run an offline agent.')
_CRITIC_ACTOR_RATIO = flags.DEFINE_integer(
    'critic_actor_update_ratio', 1, 'Critic updates per actor update.'
)
_SGD_STEPS = flags.DEFINE_integer(
    'sgd_steps', 1, 'SGD steps for online sample.'
)
_CRITIC_NETWORK = flags.DEFINE_multi_integer(
    'critic_network', [256, 256], 'Define critic architecture.'
)
_POLICY_NETWORK = flags.DEFINE_multi_integer(
    'policy_network', [256, 256, 12, 256], 'Define policy architecture.'
)
_POLICY_MODEL = flags.DEFINE_enum(
    'policy_model',
    networks.PolicyArchitectures.MLP.value,
    [e.value for e in networks.PolicyArchitectures],
    'Define policy model type.',
)
_CRITIC_MODEL = flags.DEFINE_enum(
    'critic_model',
    networks.CriticArchitectures.LNMLP.value,
    [e.value for e in networks.CriticArchitectures],
    'Define policy model type.',
)
_BC_ACTOR_LOSS = flags.DEFINE_bool(
    'bc_actor_loss', False, 'Have expert BC term in actor loss.'
)
_PRETRAIN_BC = flags.DEFINE_bool(
    'pretrain_bc', False, 'Pretrain policy from demonstrations.'
)
_BC_PRIOR = flags.DEFINE_bool(
    'bc_prior', False, 'Used pretrained BC policy as prior.'
)
_POLICY_PRETRAIN_STEPS = flags.DEFINE_integer(
    'policy_pretrain_steps', 25_000, 'Policy pretraining steps.'
)
_POLICY_PRETRAIN_LR = flags.DEFINE_float(
    'policy_pretrain_lr', 1e-3, 'Policy pretraining learning rate.'
)
_LOSS_TYPE = flags.DEFINE_enum(
    'loss_type',
    sil_config.Losses.FAITHFUL.value,
    [e.value for e in sil_config.Losses],
    'Define regression loss type.',
)
_LAYERNORM = flags.DEFINE_bool(
    'policy_layer_norm', False, 'Use layer norm for first layer of the policy.'
)
_EVAL_BC = flags.DEFINE_bool('eval_bc', False,
                             'Run evaluator of BC policy for comparison')
_EVAL_PER_VIDEO = flags.DEFINE_integer(
    'evals_per_video', 0, 'Video frequency. Disable using 0.'
)
_CHECKPOINTING = flags.DEFINE_bool(
  'checkpoint', False, 'Save models during training.'
)
_WANDB = flags.DEFINE_bool(
  'wandb', True, 'Use weights and biases logging.')

_NAME = flags.DEFINE_string('name', 'camera-ready', 'Experiment name')

def _build_experiment_config():
  """Builds an IQ-Learn experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.

  task = _ENV_NAME.value

  mode = f'{"off" if _OFFLINE_FLAG.value else "on"}line'
  name = f'iqlearn_{task}_{mode}'
  group = (f'{name}, {_NAME.value}, '
           f'ndemos={_N_DEMOS.value}, '
           f'alpha={_ENT_COEF.value}')
  wandb_kwargs = {
    'project': 'csil',
    'name': name,
    'group': group,
    'tags': ['iqlearn', task, mode, jax.default_backend()],
    'config': flags.FLAGS._flags(),
    'mode': 'online' if _WANDB.value else 'disabled',
  }

  logger_fact = experiment_logger.make_experiment_logger_factory(wandb_kwargs)

  make_env, env_spec, make_demonstrations = helpers.get_env_and_demonstrations(
      task, _N_DEMOS.value, use_sarsa=False
  )

  def environment_factory(seed: int):
    del seed
    return make_env()

  batch_size = _BATCH_SIZE.value
  seed = _SEED.value
  actor_lr = _ACTOR_LR.value

  make_demonstrations_ = lambda batchsize: make_demonstrations(batchsize, seed)

  if _PRETRAIN_BC.value:
    dataset_factory = lambda seed_: make_demonstrations(batch_size, seed_)
    policy_pretraining = [
        sil_config.PretrainingConfig(
            loss=sil_config.Losses(_LOSS_TYPE.value),
            seed=seed,
            dataset_factory=dataset_factory,
            steps=_POLICY_PRETRAIN_STEPS.value,
            learning_rate=_POLICY_PRETRAIN_LR.value,
            use_as_reference=_BC_PRIOR.value,
        ),
    ]
  else:
    policy_pretraining = None

  critic_layers = _CRITIC_NETWORK.value
  policy_layers = _POLICY_NETWORK.value
  policy_architecture = networks.PolicyArchitectures(_POLICY_MODEL.value)
  critic_architecture = networks.CriticArchitectures(_CRITIC_MODEL.value)
  use_layer_norm = _LAYERNORM.value

  def network_factory(spec: specs.EnvironmentSpec):
    return networks.make_networks(
        spec,
        policy_architecture=policy_architecture,
        critic_architecture=critic_architecture,
        critic_hidden_layer_sizes=tuple(critic_layers),
        policy_hidden_layer_sizes=tuple(policy_layers),
        layer_norm_policy=use_layer_norm,
    )

  if _ENT_COEF.value is not None and _ENT_COEF.value > 0.0:
    kwargs = {'entropy_coefficient': _ENT_COEF.value}
  else:
    kwargs = {'target_entropy': sac.target_entropy_from_env_spec(env_spec)}

  # Construct the agent.
  config = sil_config.SILConfig(
      expert_demonstration_factory=make_demonstrations_,
      imitation=sil_config.InverseSoftQConfig(
          divergence=sil_config.Divergence.CHI),
      actor_bc_loss=_BC_ACTOR_LOSS.value,
      policy_pretraining=policy_pretraining,
      discount=_DISCOUNT.value,
      critic_learning_rate=_CRITIC_LR.value,
      reward_learning_rate=_REWARD_LR.value,
      actor_learning_rate=actor_lr,
      num_sgd_steps_per_step=_SGD_STEPS.value,
      tau=_TAU.value,
      critic_actor_update_ratio=_CRITIC_ACTOR_RATIO.value,
      n_step=1,
      batch_size=batch_size,
      **kwargs,
  )

  sil_builder = builder.SILBuilder(config)

  imitation_evaluator_factory = evaluator.imitation_evaluator_factory(
      agent_config=config,
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_factory=sil_builder.make_policy,
      logger_factory=logger_fact,
  )

  evaluators = [imitation_evaluator_factory,]

  if _PRETRAIN_BC.value and _EVAL_BC.value:
    bc_evaluator_factory = evaluator.bc_evaluator_factory(
        environment_factory=environment_factory,
        network_factory=network_factory,
        policy_factory=sil_builder.make_policy,
        logger_factory=logger_fact,
    )
    evaluators += [bc_evaluator_factory,]

  if _EVAL_PER_VIDEO.value > 0:
    video_evaluator_factory = evaluator.video_evaluator_factory(
        environment_factory=environment_factory,
        network_factory=network_factory,
        policy_factory=sil_builder.make_policy,
        videos_per_eval=_EVAL_PER_VIDEO.value,
        logger_factory=logger_fact,
    )
    evaluators += [video_evaluator_factory]

  checkpoint_config = (experiments.CheckointingConfig()
                       if _CHECKPOINTING.value else None)
  if _OFFLINE_FLAG.value:
    # Note: For offline learning, the dataset needs to contain the offline and
    # expert data, so make_demonstrations isn't used and make_dataset combines
    # the two, due to how the OfflineBuilder is constructed.
    make_dataset, _ = helpers.get_offline_dataset(
        task,
        env_spec,
        _N_DEMOS.value,
        _N_OFFLINE_DATASET.value,
        use_sarsa=False,
    )
    # Offline iterator takes RNG key.
    make_dataset_ = lambda k: make_dataset(batch_size, k)
    return experiments.OfflineExperimentConfig(
        builder=sil_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        demonstration_dataset_factory=make_dataset_,
        evaluator_factories=evaluators,
        max_num_learner_steps=_N_STEPS.value,
        environment_spec=env_spec,
        seed=_SEED.value,
        logger_factory=logger_fact,
        checkpointing=checkpoint_config,
    )
  else:  # Online.
    return experiments.ExperimentConfig(
        builder=sil_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        evaluator_factories=evaluators,
        seed=_SEED.value,
        max_num_actor_steps=_N_STEPS.value,
        logger_factory=logger_fact,
        checkpointing=checkpoint_config,
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
          experiment=config, num_actors=4
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
