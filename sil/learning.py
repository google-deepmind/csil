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

"""Soft imitation learning learner implementation."""

from __future__ import annotations
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
from jax import lax
import jax.numpy as jnp
import optax

from sil import config as sil_config
from sil import networks as sil_networks
from sil import pretraining

# useful for analysis and comparing algorithms
MONITOR_BC_METRICS = False


class ImitationSample(NamedTuple):
  """For imitation learning, we require agent and demonstration experience."""

  online_sample: types.Transition
  demonstration_sample: types.Transition


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  r_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  r_params: networks_lib.Params
  key: networks_lib.PRNGKey
  bc_policy_params: Optional[networks_lib.Params] = None
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class SILLearner(acme.Learner):
  """Soft imitation learning learner."""

  _state: TrainingState

  def __init__(
      self,
      networks: sil_networks.SILNetworks,
      critic_loss_def: sil_config.CriticLossFact,
      reward_factory: sil_config.RewardFact,
      rng: jnp.ndarray,
      dataset: Iterator[ImitationSample],
      policy_optimizer: optax.GradientTransformation,
      q_optimizer: optax.GradientTransformation,
      r_optimizer: optax.GradientTransformation,
      tau: float = 0.005,
      discount: float = 0.99,
      critic_actor_update_ratio: int = 1,
      alpha_init: float = 1.0,
      alpha_learning_rate: float = 1e-3,
      entropy_coefficient: Optional[float] = None,
      target_entropy: float = 0.0,
      actor_bc_loss: bool = False,
      damping: float = 0.0,
      policy_pretraining: Optional[List[sil_config.PretrainingConfig]] = None,
      critic_pretraining: Optional[sil_config.PretrainingConfig] = None,
      counter: Optional[counting.Counter] = None,
      learner_logger: Optional[loggers.Logger] = None,
      policy_pretraining_loggers: Optional[List[loggers.Logger]] = None,
      critic_pretraining_logger: Optional[loggers.Logger] = None,
      num_sgd_steps_per_step: int = 1,
  ):
    """Initialize the soft imitation learning learner.

    Args:
      networks: SIL networks
      critic_loss_def: loss function definition for critic
      reward_factory: create implicit or explicit reward functions
      rng: a key for random number generation.
      dataset: an iterator over demonstrations and online data.
      policy_optimizer: the policy optimizer.
      q_optimizer: the Q-function optimizer.
      r_optimizer: the reward function optimizer.
      tau: target smoothing coefficient.
      discount: discount to use for TD updates.
      critic_actor_update_ratio: critic updates per single actor update.
      alpha_init:
      alpha_learning_rate:
      entropy_coefficient: coefficient applied to the entropy bonus. If None, an
        adaptative coefficient will be used.
      target_entropy: Used to normalize entropy. Only used when
        entropy_coefficient is None.
      actor_bc_loss: add auxiliary BC term to actor objective (unused)
      damping: damping of KL constraint
      policy_pretraining: Optional config for pretraining policy
      critic_pretraining: Optional config for pretraining critic
      counter: counter object used to keep track of steps.
      learner_logger: logger object to be used by learner.
      policy_pretraining_loggers: logger objects to be used by the policy pretraining.
      critic_pretraining_logger: logger object to be used by critic pretraining.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """

    adaptive_entropy_coefficient = entropy_coefficient is None
    kl_bound = jnp.abs(target_entropy)

    if adaptive_entropy_coefficient:
      # Alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      # Invert softplus to initial virtual value.
      if alpha_init > 4.:
        virtual_alpha = jnp.asarray(alpha_init, dtype=jnp.float32)
      else:  # safely invert softplus
        virtual_alpha = jnp.log(
            jnp.exp(jnp.asarray(alpha_init, dtype=jnp.float32)) - 1.0
        )
      alpha_optimizer = optax.sgd(learning_rate=alpha_learning_rate)
      alpha_optimizer_state = alpha_optimizer.init(virtual_alpha)
    else:
      if target_entropy:
        raise ValueError(
            'target_entropy should not be set when '
            'entropy_coefficient is provided'
        )

    def make_initial_state(
        key: networks_lib.PRNGKey,
    ) -> Tuple[TrainingState, Optional[networks_lib.Params], bool]:
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)

      # In the online setting we pretrain the policy against the demonstrations.
      # In the offfline setting we pretrain the policy against the dataset
      # and demonstrations.
      # In both cases, we use the last trained policy for the CSIL reward,
      # and in the offline case, we use the first policy as the 'BC' policy
      # to stay close to the data.
      bc_policy_params = []
      use_pretrained_prior = False
      if policy_pretraining:
        for i, pt in enumerate(policy_pretraining):
          use_pretrained_prior = use_pretrained_prior or pt.use_as_reference
          if not bc_policy_params:
            policy_ = networks_lib.FeedForwardNetwork(
                networks.bc_policy_network.init,
                networks.bc_policy_network.apply,
            )
          else:
            policy_ = networks_lib.FeedForwardNetwork(
                lambda key: bc_policy_params[0],
                networks.bc_policy_network.apply,
            )
          params = pretraining.behavioural_cloning_pretraining(
              loss=pt.loss,
              seed=pt.seed,
              env_spec=networks.environment_specs,
              dataset_factory=pt.dataset_factory,
              policy=policy_,
              learning_rate=pt.learning_rate,
              num_steps=pt.steps,
              logger=policy_pretraining_loggers[i],
              name=f'{i}',
          )
          bc_policy_params += [params,]
      else:
        bc_policy_params = [None]
      # While IQ-Learn and P2IL use policy pretraining for the policy, CSIL can
      # use it only for the reward initialization.
      policy_match = (
          networks.policy_architecture == networks.bc_policy_architecture
      )

      if policy_match and policy_pretraining:
        policy_params = bc_policy_params[0].copy()
      else:
        policy_params = networks.policy_network.init(key_policy)

      policy_optimizer_state = policy_optimizer.init(policy_params)

      if networks.reward_policy_coherence and bc_policy_params[-1]:
        r_params = bc_policy_params[-1].copy()
      else:
        r_params = networks.reward_network.init(key_q)
        # Share encoder with policy if present.
        r_params = sil_networks.update_encoder(r_params, policy_params)

      r_optimizer_state = r_optimizer.init(r_params)

      if critic_pretraining is not None:
        critic_ = networks_lib.FeedForwardNetwork(
            networks.critic_network.init, networks.critic_network.apply
        )
        reward_ = networks_lib.FeedForwardNetwork(
            networks.reward_network.init, networks.reward_network.apply
        )
        policy_ = networks_lib.FeedForwardNetwork(
            networks.policy_network.init, networks.policy_network.apply
        )
        critic_params = critic_.init(key_q)
        critic_params = sil_networks.update_encoder(
            critic_params, policy_params)
        critic_params = pretraining.critic_pretraining(
            seed=critic_pretraining.seed,
            dataset_factory=critic_pretraining.dataset_factory,
            critic=critic_,
            critic_params=critic_params,
            reward=reward_,
            reward_params=r_params,
            discount_factor=discount,
            num_steps=critic_pretraining.steps,
            learning_rate=critic_pretraining.learning_rate,
            logger=critic_pretraining_logger,
        )
      else:
        critic_params = networks.critic_network.init(key_q)
        # Share encoder with policy if present.
        critic_params = sil_networks.update_encoder(
            critic_params, policy_params)

      q_optimizer_state = q_optimizer.init(critic_params)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          r_optimizer_state=r_optimizer_state,
          policy_params=policy_params,
          q_params=critic_params,
          target_q_params=critic_params,
          r_params=r_params,
          bc_policy_params=bc_policy_params[-1],
          key=key,
      )

      if adaptive_entropy_coefficient:
        state = state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=virtual_alpha,
        )
      return state, bc_policy_params[-1], use_pretrained_prior

    # Create initial state.
    self._state, bc_policy_params, use_policy_prior = make_initial_state(rng)

    if use_policy_prior:
      assert bc_policy_params is not None

    def alpha_loss(
        virtual_alpha: jnp.ndarray,
        policy_params: networks_lib.Params,
        transitions: types.Transition,
        key: networks_lib.PRNGKey,
    ) -> jnp.ndarray:
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation
      )
      action = dist_params.sample(seed=key)
      log_prob = networks.log_prob(dist_params, action)
      alpha = jax.nn.softplus(virtual_alpha)
      if use_policy_prior:  # bc_policy_params are not None
        dist_bc = networks.bc_policy_network.apply(
            bc_policy_params, transitions.observation
        )
        prior_log_prob = networks.log_prob(dist_bc, action)
        kl = (log_prob - prior_log_prob).mean()
        # KL constraint.
        constraint = jax.lax.stop_gradient(kl_bound - kl)
        # Zero if kl < kl_bound, negative if violated.
        # We want temp to go zero if kl not violated, so don't clip here.
        loss = constraint
        # Do gradient ascent so invert sign w.r.t. actor loss term.
        alpha_loss = alpha * loss
      else:
        alpha_loss = (
            alpha * jax.lax.stop_gradient(-log_prob - target_entropy).mean()
        )
      return alpha_loss

    def critic_loss(
        q_params: networks_lib.Params,
        r_params: networks_lib.Params,
        policy_params: networks_lib.Params,
        target_q_params: networks_lib.Params,
        alpha: jnp.ndarray,
        demonstration_transitions: types.Transition,
        online_transitions: types.Transition,
        key: networks_lib.PRNGKey,
    ):
      # The key aspect of soft imitation learning is the critic objective and
      # reward. We obtain these from factories defined in the config.
      def state_action_reward_fn(state, action):
        return jnp.ravel(networks.reward_network.apply(r_params, state, action))

      def state_action_value_fn(state, action):
        return networks.critic_network.apply(
            q_params, state, action).min(axis=-1)  # reduce via min even for 1D

      def _state_value_fn(state, critic_params, policy_key):
        # SAC's soft value function, see Equation 3 of
        # https://arxiv.org/pdf/1812.05905.pdf.
        action_dist = networks.policy_network.apply(policy_params, state)
        action = action_dist.sample(seed=policy_key)
        policy_log_prob = networks.log_prob(action_dist, action)
        if use_policy_prior:  # bc_policy_params have been trained
          prior_log_prob = networks.bc_policy_network.apply(
              bc_policy_params, state
          ).log_prob(action)
        else:
          prior_log_prob = networks.log_prob_prior(action)
        q = networks.critic_network.apply(
            critic_params, state, action).min(axis=-1)
        return q - alpha * (policy_log_prob - prior_log_prob)

      def state_value_fn(
          state: jnp.ndarray, key: jax.Array
      ) -> jnp.ndarray:
        return _state_value_fn(state, q_params, key)

      def target_state_value_fn(
          state: jnp.ndarray, key: jax.Array
      ) -> jnp.ndarray:
        return lax.stop_gradient(_state_value_fn(state, target_q_params, key))

      reward_fn = reward_factory(
          state_action_reward_fn,
          state_action_value_fn,
          target_state_value_fn,
          discount,
      )

      critic_loss, metrics = critic_loss_def(
          reward_fn,
          state_action_value_fn,
          state_value_fn,
          target_state_value_fn,
          discount,
          demonstration_transitions,
          online_transitions,
          key,
      )
      return critic_loss, metrics

    def actor_loss(
        policy_params: networks_lib.Params,
        q_params: networks_lib.Params,
        alpha: jnp.ndarray,
        demonstration_transitions: types.Transition,
        online_transitions: types.Transition,
        key: networks_lib.PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict[str, float | jnp.Array]]:

      def action_sample(
          observation: jnp.ndarray,
          action_key: jax.Array,
      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        dist = networks.policy_network.apply(policy_params, observation)
        sample = dist.sample(seed=action_key)
        log_prob = networks.log_prob(dist, sample)
        return dist.mode(), sample, log_prob, dist

      observation = sil_config.concatenate(
          demonstration_transitions.observation, online_transitions.observation
      )
      expert_mode, expert_action, expert_log_prob, expert_dist = action_sample(
          demonstration_transitions.observation, key
      )
      online_mode, online_action, online_log_prob, _ = action_sample(
          online_transitions.observation, key
      )

      action = sil_config.concatenate(expert_action, online_action)
      action_mode = sil_config.concatenate(expert_mode, online_mode)
      log_prob = sil_config.concatenate(expert_log_prob, online_log_prob)
      prior_log_prob = networks.log_prob_prior(action)
      # Use min as reducer in case we use two or one critic functions.
      q = networks.critic_network.apply(
          q_params, observation, action).min(axis=-1)
      q_mode = networks.critic_network.apply(
          q_params, observation, action_mode).min(axis=-1)

      if use_policy_prior:
        dist_bc = networks.bc_policy_network.apply(
            bc_policy_params, observation
        )
        prior_log_prob = networks.log_prob(dist_bc, action)
        kl = (log_prob - prior_log_prob).mean()
        constraint = (kl_bound - kl).sum()  # Sum to reduce to scalar.
        clipped_constraint = jnp.clip(constraint, a_max=0.0)
        d = damping * clipped_constraint ** 2
        # Constraint is <= 0, so negate for loss.
        entropy_reg = -alpha * constraint + d
      else:  # Vanilla maximum entropy regularization with uniform prior.
        d = 0.0
        kl = (log_prob - prior_log_prob).mean()
        constraint = kl
        clipped_constraint = 0.0
        entropy_reg = alpha * kl

      actor_loss = entropy_reg - q.mean()

      if actor_bc_loss:

        # For SAC's tanh policy, the minimizing modal MSE and maximizing
        # loglikelihood do not appear to be mutually guaranteed, so we optimize
        # for both.
        # Incorporate BC MSE loss from TD3+BC.
        # https://arxiv.org/abs/2106.06860
        expert_se = (expert_mode - demonstration_transitions.action) ** 2
        bc_loss_mean = 0.5 * expert_se.mean() * jnp.abs(q).mean()

        # Also incorporate a log-likelihood, which should be similar in value to
        # the entropy as they are constructed in similar ways, so use alpha to
        # weight. This is like maximum likelihood with an entropy bonus.
        # See https://proceedings.mlr.press/v97/jacq19a/jacq19a.pdf Section 5.2.
        expert_demo_log_prob = networks.log_prob(
            expert_dist, demonstration_transitions.action
        )
        bc_loss_mean += -alpha * expert_demo_log_prob.mean()

        actor_loss += bc_loss_mean

      metrics = {
          'actor_q': q.mean(),
          'actor_q_mode': q_mode.mean(),
          'actor_entropy_bonus': (alpha * log_prob).mean(),
          'actor_kl': kl,
          'kl_bound': kl_bound,
          'constraint': constraint,
          'clipped_constraint': clipped_constraint,
          'entropy_reg': entropy_reg,
          'prior_log_prob': prior_log_prob.mean(),
          'policy_log_prob': log_prob.mean(),
          'damping': d,
      }
      return actor_loss, metrics

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, argnums=[0, 1], has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

    def update_step(
        state: TrainingState,
        sample: ImitationSample,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
      # Update temperature, actor and critic.
      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      alpha_grads = None
      alpha_loss = None
      if adaptive_entropy_coefficient:
        transition = sil_config.concatenate_transitions(
            sample.online_sample, sample.demonstration_sample
        )
        alpha_loss, alpha_grads = alpha_grad(
            state.alpha_params, state.policy_params, transition, key_alpha
        )
        alpha = jax.nn.softplus(state.alpha_params)
      else:
        alpha = entropy_coefficient

      # Update critic (and reward).

      q_params = state.q_params
      r_params = state.r_params
      target_q_params = state.target_q_params
      critic_loss = None
      critic_grads = None
      critic_loss_metrics = None
      q_optimizer_state = None
      r_optimizer_state = None
      for _ in range(critic_actor_update_ratio):
        critic_losses, grads = critic_grad(
            q_params,
            r_params,
            state.policy_params,
            target_q_params,
            alpha,
            sample.demonstration_sample,
            sample.online_sample,
            key_critic,
        )
        critic_loss, critic_loss_metrics = critic_losses
        critic_grads, reward_grads = grads

        # Apply critic gradients.
        critic_update, q_optimizer_state = q_optimizer.update(
            critic_grads, state.q_optimizer_state, q_params
        )
        q_params = optax.apply_updates(q_params, critic_update)

        reward_update, r_optimizer_state = r_optimizer.update(
            reward_grads, state.r_optimizer_state, r_params
        )
        r_params = optax.apply_updates(r_params, reward_update)

      target_q_params = jax.tree_map(
          lambda x, y: x * (1 - tau) + y * tau, target_q_params, q_params
      )

      # Update actor.
      actor_losses, actor_grads = actor_grad(
          state.policy_params,
          q_params,
          alpha,
          sample.demonstration_sample,
          sample.online_sample,
          key_actor,
      )
      actor_loss, actor_loss_metrics = actor_losses
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      metrics = {
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
          'critic_grad_norm': optax.global_norm(critic_grads),
          'actor_grad_norm': optax.global_norm(actor_grads),
      }

      metrics.update(critic_loss_metrics)
      metrics.update(actor_loss_metrics)

      if MONITOR_BC_METRICS:
        # During training, expert actions should become / stay high likelihood.
        expert_action_dist = networks.policy_network.apply(
            policy_params, sample.demonstration_sample.observation
        )
        samp = expert_action_dist.sample(seed=key)
        expert_ent_approx = -networks.log_prob(expert_action_dist, samp).mean()
        expert_llhs = networks.log_prob(
            expert_action_dist, sample.demonstration_sample.action
        )
        expert_se = (
            expert_action_dist.mode() - sample.demonstration_sample.action
        ) ** 2
        online_action_dist = networks.policy_network.apply(
            policy_params, sample.online_sample.observation
        )
        samp = online_action_dist.sample(seed=key)
        online_ent_approx = -networks.log_prob(online_action_dist, samp).mean()
        online_llh = networks.log_prob(
            online_action_dist, sample.online_sample.action
        ).mean()
        online_se = (online_action_dist.mode() - sample.online_sample.action) ** 2

        metrics.update({
            'expert_llh_mean': expert_llhs.mean(),
            'expert_llh_max': expert_llhs.max(),
            'expert_llh_min': expert_llhs.min(),
            'expert_mse': expert_se.mean(),
            'online_llh': online_llh,
            'online_mse': online_se.mean(),
            'expert_ent': expert_ent_approx,
            'online_ent': online_ent_approx,
        })

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          r_optimizer_state=r_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=target_q_params,
          r_params=r_params,
          bc_policy_params=state.bc_policy_params,
          key=key,
      )
      if adaptive_entropy_coefficient:
        # Apply alpha gradients.
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        metrics.update({
            'alpha_loss': alpha_loss,
            'alpha': jax.nn.softplus(alpha_params),
        })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)

      metrics['rewards_mean'] = jnp.mean(
          jnp.abs(jnp.mean(sample.online_sample.reward, axis=0))
      )
      metrics['rewards_std'] = jnp.std(sample.online_sample.reward, axis=0)

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = learner_logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())
    self._num_sgd_steps_per_step = num_sgd_steps_per_step

    # Iterator on demonstration transitions.
    self._iterator = dataset

    # Use the JIT compiler.
    self._update_step = jax.jit(update_step)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):

    metrics = {}
    # Update temperature, actor and critic.
    for _ in range(self._num_sgd_steps_per_step):
      sample = next(self._iterator)
      self._state, metrics = self._update_step(self._state, sample)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[Any]:
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.q_params,
        'reward': self._state.r_params,
        'bc_policy_params': self._state.bc_policy_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
