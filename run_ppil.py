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

"""Example running proximal point imitation learning on continuous control tasks."""

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
    'eval_every', 5_000, 'How often to evaluate for local runs.'
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
_DISCOUNT = flags.DEFINE_float('discount', 0.99, 'Discount factor')
_ACTOR_LR = flags.DEFINE_float('actor_lr', 3e-5, 'Actor learning rate.')
_CRITIC_LR = flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate.')
_REWARD_LR = flags.DEFINE_float('reward_lr', 3e-4, 'Reward learning rate.')
_TAU = flags.DEFINE_float(
    'tau',
    0.005,
    (
        'Target network exponential smoothing weight.'
        '1. = no update, 0, = no smoothing.'
    ),
)
_ENT_COEF = flags.DEFINE_float(
    'entropy_coefficient',
    None,
    'Entropy coefficient. Becomes adaptive if None.',
)
_CRITIC_ACTOR_RATIO = flags.DEFINE_integer(
    'critic_actor_update_ratio', 20, 'Critic updates per actor update.'
)
_SGD_STEPS = flags.DEFINE_integer(
    'sgd_steps', 1, 'SGD steps for online sample.'
)
_CRITIC_NETWORK = flags.DEFINE_multi_integer(
    'critic_network', [256, 256], 'Define critic architecture.'
)
_REWARD_NETWORK = flags.DEFINE_multi_integer(
    'reward_network', [256, 256], 'Define reward architecture.'
)
_POLICY_NETWORK = flags.DEFINE_multi_integer(
    'policy_network', [256, 256], 'Define policy architecture.'
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
    'Define critic model type.',
)
_REWARD_MODEL = flags.DEFINE_enum(
    'reward_model',
    networks.RewardArchitectures.LNMLP.value,
    [e.value for e in networks.RewardArchitectures],
    'Define reward model type.',
)
_BET = flags.DEFINE_float(
    'bellman_error_temp', 0.08, 'Temperature of logistic Bellman error term.'
)
_CSIL_ALPHA = flags.DEFINE_float('csil_alpha', None, 'CSIL reward temperature.')
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
_OFFLINE_FLAG = flags.DEFINE_bool('offline', False, 'Run an offline agent.')
_EVAL_BC = flags.DEFINE_bool(
  'eval_bc', False, 'Run evaluator of BC policy for comparison')
_CHECKPOINTING = flags.DEFINE_bool(
  'checkpoint', False, 'Save models during training.'
)
_WANDB = flags.DEFINE_bool(
  'wandb', True, 'Use weights and biases logging.'
)
_NAME = flags.DEFINE_string('name', 'camera-ready', 'Experiment name')

def build_experiment_config():
  """Builds a P2IL experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.

  task = _ENV_NAME.value

  mode = f'{"off" if _OFFLINE_FLAG.value else "on"}line'
  name = f'ppil_{task}_{mode}'
  group = (f'{name}, {_NAME.value}, '
           f'ndemos={_N_DEMONSTRATIONS.value}, '
           f'alpha={_ENT_COEF.value}')
  wandb_kwargs = {
    'project': 'csil',
    'name': name,
    'group': group,
    'tags': ['ppil', task, mode, jax.default_backend()],
    'mode': 'online' if _WANDB.value else 'disabled',
  }

  logger_fact = experiment_logger.make_experiment_logger_factory(wandb_kwargs)

  make_env, env_spec, make_demonstrations = helpers.get_env_and_demonstrations(
      task, _N_DEMONSTRATIONS.value
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

  critic_layers, reward_layers, policy_layers = (
      _CRITIC_NETWORK.value,
      _REWARD_NETWORK.value,
      _POLICY_NETWORK.value,
  )

  policy_architecture = networks.PolicyArchitectures(_POLICY_MODEL.value)
  critic_architecture = networks.CriticArchitectures(_CRITIC_MODEL.value)
  reward_architecture = networks.RewardArchitectures(_REWARD_MODEL.value)
  csil_alpha = _CSIL_ALPHA.value

  def network_factory(spec: specs.EnvironmentSpec):
    return networks.make_networks(
        spec,
        policy_architecture=policy_architecture,
        critic_architecture=critic_architecture,
        reward_architecture=reward_architecture,
        critic_hidden_layer_sizes=tuple(critic_layers),
        reward_hidden_layer_sizes=tuple(reward_layers),
        policy_hidden_layer_sizes=tuple(policy_layers),
        reward_policy_coherence_alpha=csil_alpha,
    )

  if _ENT_COEF.value is not None and _ENT_COEF.value > 0.0:
    kwargs = {'entropy_coefficient': _ENT_COEF.value}
  else:
    kwargs = {'target_entropy': sac.target_entropy_from_env_spec(env_spec)}

  # Construct the agent.
  config = sil_config.SILConfig(
      imitation=sil_config.ProximalPointConfig(
          bellman_error_temperature=_BET.value),
      actor_bc_loss=_BC_ACTOR_LOSS.value,
      policy_pretraining=policy_pretraining,
      expert_demonstration_factory=make_demonstrations_,
      discount=_DISCOUNT.value,
      critic_learning_rate=_CRITIC_LR.value,
      reward_learning_rate=_REWARD_LR.value,
      actor_learning_rate=actor_lr,
      num_sgd_steps_per_step=_SGD_STEPS.value,
      critic_actor_update_ratio=_CRITIC_ACTOR_RATIO.value,
      n_step=1,
      tau=_TAU.value,
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
    evaluators  += [bc_evaluator_factory,]

  checkpoint_config = (experiments.CheckointingConfig()
                       if _CHECKPOINTING.value else None)
  if _OFFLINE_FLAG.value:
    make_dataset, _ = helpers.get_offline_dataset(
        task, env_spec, _N_DEMONSTRATIONS.value, _N_OFFLINE_DATASET.value
    )
    # The offline runner needs a random seed for the dataset.
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
  else:
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
  config = build_experiment_config()
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
