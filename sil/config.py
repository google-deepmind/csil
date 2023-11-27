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

"""Soft imitation learning types, configurations and hyperparameters.

Useful resources for these implementations:
    IQ-Learn: https://arxiv.org/abs/2106.12142
              https://github.com/Div99/IQ-Learn
    P^2IL:    https://arxiv.org/abs/2209.10968
              https://github.com/lviano/P2IL
"""
import abc
import dataclasses
import enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from acme import types
from acme.agents.jax.sac import config as sac_config
import jax
from jax import lax
from jax import random
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import tree

# Custom function and return typing for signature brevity.
# Generalized reward function, including discounting.
StateActionNextStateFunc = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]
# State-action critics.
StateActionFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
# I.e. function regularizers.
StateFunc = Callable[[jnp.ndarray], jnp.ndarray]
# Soft (i.e. stochastic) value function.
StochStateFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
# Losses and factories for defined here for brevity.
AuxLoss = Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
CriticLossFact = Callable[
    [
        StateActionNextStateFunc,
        StateActionFunc,
        StochStateFunc,
        StochStateFunc,
        float,
        types.Transition,
        types.Transition,
        jax.Array,
    ],
    AuxLoss,
]
RewardFact = Callable[
    [StateActionFunc, StateActionFunc, StochStateFunc, float],
    StateActionNextStateFunc,
]
DemonstrationFact = Callable[..., Iterator[types.Transition]]
ConcatenateFact = Callable[
    [types.Transition, types.Transition], types.Transition
]


# Used to bound policy log-likelihoods via the variance.
MIN_VAR = 1e-5

# Maximum reward for CSIL.
# By having a lower bound on the variance, and setting alpha to be
# (1 - \gamma) / dim_a, we can bound the maximum discounted return.
MAX_REWARD = -0.5 * np.log(MIN_VAR) + np.log(2.)


class Losses(enum.Enum):
  FAITHFUL = "faithful"
  DBFAITHFUL = "dbfaithful"
  MSE = "mse"
  NLLH = "nllh"

  def __str__(self) -> str:
    return self.value


# Soft inverse Q learning has different objectives based on divergence choice
# between expert and policy occupancy measure.
class Divergence(enum.Enum):
  FORWARD_KL = 0
  REVERSE_KL = 1
  REVERSE_KL_DUAL = 2
  REVERSE_KL_UNBIASED = 3
  HELLINGER = 4
  JS = 5
  CHI = 6
  TOTAL_VARIATION = 7


# Each f divergence can be written as
# D_f[P || Q] = \sup_g E_P[g(x)] - E_Q[f^*(g(x))]
# where f^* is the convex conjugate of f.
# For the objective, we require a function \phi(x) = -f^*(-x).
# These functions implement the \phi(x) for a given divergence.
# See Table 4 of Garg et al. (2021) and Ho et al. (2016) for more details.
_forward_kl: StateFunc = lambda r: 1.0 + jnp.log(r)
_reverse_kl: StateFunc = lambda r: jnp.exp(-r - 1.0)
_reverse_kl_dual: StateFunc = lambda r: jnn.softmax(-r) * r.shape[0]
_reverse_kl_unbiased: StateFunc = lambda r: jnp.exp(-r)
_hellinger: StateFunc = lambda r: 1.0 / (1.0 + r) ** 2
_chi: StateFunc = lambda r: r - r**2 / 2.0
_total_variation: StateFunc = lambda r: r
_js: StateFunc = lambda r: jnp.log(2.0 - jnp.exp(-r))

DIVERGENCE_REGULARIZERS = {
    Divergence.FORWARD_KL: _forward_kl,
    Divergence.REVERSE_KL: _reverse_kl,
    Divergence.REVERSE_KL_DUAL: _reverse_kl_dual,
    Divergence.REVERSE_KL_UNBIASED: _reverse_kl_unbiased,
    Divergence.HELLINGER: _hellinger,
    Divergence.CHI: _chi,
    Divergence.TOTAL_VARIATION: _total_variation,
    Divergence.JS: _js,
}


def concatenate(x: Any, y: Any) -> Any:
  return tree.map_structure(lambda x, y: jnp.concatenate((x, y), axis=0), x, y)


def concatenate_transitions(
    x: types.Transition, y: types.Transition
) -> types.Transition:
  fields = ["observation", "action", "reward", "discount", "next_observation"]
  return types.Transition(
      *[concatenate(getattr(x, field), getattr(y, field)) for field in fields]
  )


@dataclasses.dataclass
class SoftImitationConfig(abc.ABC):
  """Abstact base class for soft imitation learning."""

  @abc.abstractmethod
  def critic_loss_factory(self) -> CriticLossFact:
    """Define the critic loss based on the algorithm."""

  @abc.abstractmethod
  def reward_factory(self) -> RewardFact:
    """Define the reward based on the algorithm, i.e. implicit vs explicit."""


@dataclasses.dataclass
class InverseSoftQConfig(SoftImitationConfig):
  """Configuration for the IQ-Learn algorithm."""

  # Defines divergence-derived function regularization.
  divergence: Divergence = Divergence.CHI

  def critic_loss_factory(self) -> CriticLossFact:
    """Generate critic objective from hyperparameters."""
    regularizer = DIVERGENCE_REGULARIZERS[self.divergence]

    def objective(
        reward_fn: StateActionNextStateFunc,
        state_action_value_fn: StateActionFunc,
        value_fn: StochStateFunc,
        target_value_fn: StochStateFunc,
        discount: float,
        demonstration_transitions: types.Transition,
        online_transitions: types.Transition,
        key: jax.Array,
    ) -> AuxLoss:
      """See Equation 10 of Garg et al. (2021) for reference."""
      del target_value_fn
      key_er, key_v, key_or = random.split(key, 3)
      expert_reward = reward_fn(
          demonstration_transitions.observation,
          demonstration_transitions.action,
          demonstration_transitions.next_observation,
          demonstration_transitions.discount,
          key_er,
      )
      phi_grad = regularizer(expert_reward).mean()
      expert_loss = -phi_grad.mean()

      # This is denoted the value of the initial state distribution in the paper
      # and codebase, but in practice the distribution is replaced with the
      # demonstration distribution. In practice this term ensures the Q
      # function is maximized at the demonstration data since here we are
      # minimizing the value for action sampled around the optimal expert.
      value_reg = (1 - discount) * value_fn(
          demonstration_transitions.observation, key_v
      ).mean()

      online_reward = reward_fn(
          online_transitions.observation,
          online_transitions.action,
          online_transitions.next_observation,
          online_transitions.discount,
          key_or,
      )
      # See code implementation IQ-Learn/iq_learn/iq.py
      agent_loss = 0.5 * (online_reward**2).mean()

      metrics = {
          "expert_reward": expert_reward.mean(),
          "online_reward": online_reward.mean(),
          "value_reg": value_reg,
          "expert_loss": expert_loss,
          "online_reg": agent_loss,
      }
      return expert_loss + value_reg + agent_loss, metrics

    return objective

  def reward_factory(self) -> RewardFact:
    def reward_factory_(
        state_action_reward: StateActionFunc,
        state_action_value_function: StateActionFunc,
        state_value_function: StochStateFunc,
        discount_factor: float,
    ) -> StateActionNextStateFunc:
      del state_action_reward

      def reward_fn(
          state: jnp.ndarray,
          action: jnp.ndarray,
          next_state: jnp.ndarray,
          discount: jnp.ndarray,
          value_key: jnp.ndarray,
      ) -> jnp.ndarray:
        q = state_action_value_function(state, action)
        future_v = discount * state_value_function(next_state, value_key)
        return q - discount_factor * future_v

      return reward_fn

    return reward_factory_


@dataclasses.dataclass
class ProximalPointConfig(SoftImitationConfig):
  """Configuration for the proximal point imitation learning algorithm."""

  bellman_error_temperature: float = 1.0

  def critic_loss_factory(self) -> CriticLossFact:
    """Generate critic objective from hyperparameters.

    P^2IL's objective consists of four terms.
    Optimizing the experts rewards, minimizing the logistic Bellman error,
    optimizing the initial state value using a proxy form to improve the value
    function, and regularizing the reward function with a squared penality.

    Returns:
      Function that return the critic loss function.
    """
    alpha = self.bellman_error_temperature

    def objective(
        reward_fn: StateActionNextStateFunc,
        state_action_value_fn: StateActionFunc,
        value_fn: StochStateFunc,
        target_value_fn: StochStateFunc,
        discount: float,
        demonstration_transitions: types.Transition,
        online_transitions: types.Transition,
        key: jax.Array,
    ) -> AuxLoss:
      """See Equation 67 and Theorem 6 of Viano et al. (2022) for reference."""
      key_cr, key_er, key_v, key_fv, key_or = random.split(key, 5)
      expert_reward = reward_fn(
          demonstration_transitions.observation,
          demonstration_transitions.action,
          demonstration_transitions.next_observation,
          demonstration_transitions.discount,
          key_er,
      )
      expert_reward_mean = expert_reward.mean()
      expert_q = state_action_value_fn(
          demonstration_transitions.observation,
          demonstration_transitions.action,
      )
      expert_v = value_fn(demonstration_transitions.observation, key_v)
      expert_next_v = value_fn(
          demonstration_transitions.next_observation, key_fv
      )
      expert_d = discount * demonstration_transitions.discount
      expert_imp_r_mean = (expert_q - expert_d * expert_next_v).mean()

      online_r = reward_fn(
          online_transitions.observation,
          online_transitions.action,
          online_transitions.next_observation,
          online_transitions.discount,
          key_or,
      )
      online_q = state_action_value_fn(
          online_transitions.observation, online_transitions.action
      )
      online_v = value_fn(online_transitions.observation, key_v)
      online_d = online_transitions.discount
      combined_transition = concatenate_transitions(
          demonstration_transitions, online_transitions
      )
      combined_reward = reward_fn(
          combined_transition.observation,
          combined_transition.action,
          combined_transition.next_observation,
          online_transitions.discount,
          key_cr,
      )
      combined_q = state_action_value_fn(
          combined_transition.observation, combined_transition.action
      )
      # The Bellman equation needs the target value function for stability,
      # while the regularization shapes the current value function.
      combined_next_target_v = target_value_fn(
          combined_transition.next_observation, key_fv
      )

      # In theory, the reward function should be jointly optimized in the
      # Bellman equation to minimize the on-policy rewards, however, empirically
      # this produced worse results as the Bellman equation is harder to
      # minimize.
      discount_ = discount * combined_transition.discount
      combined_q_target = combined_reward + discount_ * combined_next_target_v
      bellman_error = lax.stop_gradient(combined_q_target) - combined_q
      # In theory, the Bellman error should not be negative as the Q and V
      # function should roughly track in magnitude, but in practice this was not
      # always the case.
      bellman_error = jnp.abs(bellman_error)
      # Self-normalized importance sampling weights z in Theorem 6 of
      # Vivano et al. (2022), used in logsumexp objective.
      log_weights = lax.stop_gradient(alpha * bellman_error)
      norm_weights = jnn.softmax(log_weights)
      ess = 1.0 / (norm_weights**2).sum()  # effective sample size
      is_bellman_error = jnp.einsum("b,b->", norm_weights, bellman_error)
      sq_bellman_error = (bellman_error**2).mean()
      rms_bellman_error = jnp.sqrt(sq_bellman_error)

      apprenticeship_loss = -expert_reward.mean() + jnp.einsum(
          "b,b->", norm_weights, combined_reward
      )

      # This is denoted the value of the initial state distribution in the paper
      # and codebase, but in practice the distribution is replaced with the
      # demonstration distribution. In practice this term ensures the Q
      # function is maximized at the demonstration data since here we are
      # minimizing the value for action sampled around the optimal expert.
      value_reg = (1.0 - discount) * expert_v.mean()

      # P^2IL uses IQ-Learn Chi^2-based regularization in practice.
      expert_r_mean = expert_reward_mean
      function_reg = 0.5 * (combined_reward**2).mean()
      metrics = {
          "apprenticeship_loss": apprenticeship_loss,
          "expert_reward": expert_reward.mean(),
          "expert_reward_implicit": expert_imp_r_mean,
          "expert_reward_combined": expert_r_mean,
          "online_reward": online_r.mean(),
          "value_reg": value_reg,
          "function_reg": function_reg,
          "sq_bellman_error": sq_bellman_error,
          "is_bellman_error": is_bellman_error,
          "ess": ess,
      }
      return (
          -expert_r_mean + is_bellman_error + value_reg + function_reg,
          metrics,
      )

    return objective

  def reward_factory(self) -> RewardFact:
    """P^2IL's reward function is a straightforwad MLP."""

    def reward_factory_(
        state_action_reward: StateActionFunc,
        state_action_value_function: StateActionFunc,
        state_value_function: StochStateFunc,
        discount_factor: float,
    ) -> StateActionNextStateFunc:
      del state_action_value_function, state_value_function, discount_factor

      def reward_fn(
          state: jnp.ndarray,
          action: jnp.ndarray,
          next_state: jnp.ndarray,
          discount: jnp.ndarray,
          value_key: jnp.ndarray,
      ) -> jnp.ndarray:
        del next_state, discount, value_key
        return state_action_reward(state, action)

      return reward_fn

    return reward_factory_


@dataclasses.dataclass
class CoherentConfig(SoftImitationConfig):
  """Coherent soft imitation learning."""

  alpha: float # temperature used in the coherent reward
  reward_scaling: float = 1.0
  scale_factor: float = 1.0  # Scaling of online reward regularization.
  grad_norm_sf: float = 1.0  # Critic action Jacobian regularization.
  refine_reward: bool = True
  negative_reward: bool = False

  def critic_loss_factory(self) -> CriticLossFact:
    def objective(
        reward_fn: StateActionNextStateFunc,
        state_action_value_fn: StateActionFunc,
        value_fn: StochStateFunc,
        target_value_fn: StochStateFunc,
        discount: float,
        demonstration_transitions: types.Transition,
        online_transitions: types.Transition,
        key: jax.Array,
    ) -> AuxLoss:
      key_er, key_fv, key_or = random.split(key, 3)
      combined_transition = concatenate_transitions(
          demonstration_transitions, online_transitions
      )

      online_reward = reward_fn(
          online_transitions.observation,
          online_transitions.action,
          online_transitions.next_observation,
          online_transitions.discount,
          key_or,
      )

      expert_reward = reward_fn(
          demonstration_transitions.observation,
          demonstration_transitions.action,
          demonstration_transitions.next_observation,
          online_transitions.discount,
          key_er,
      )

      reward = lax.stop_gradient(jnp.concatenate(
        (expert_reward, online_reward), axis=0))

      state_action_value = state_action_value_fn(
          combined_transition.observation, combined_transition.action
      )
      future_value = target_value_fn(
          combined_transition.next_observation, key_fv
      )
      # Use knowledge to bound rogue Q values.
      max_value = 0. if self.negative_reward else MAX_REWARD / (1. - discount)
      future_value = jnp.clip(future_value, a_max=max_value)

      discount_ = discount * combined_transition.discount

      target_state_action_value = (reward + discount_ * future_value)
      bellman_error = state_action_value - target_state_action_value

      sbe = (bellman_error**2).mean()
      be = bellman_error.mean()

      # The mean expert reward corresponse to maximizing BC likelihood + const.,
      # minimizing the online reward corresponds to minimizing KL to prior.
      # Use an unbiased KL estimator that can never be negative.al
      expert_reward_mean = expert_reward.mean()  # + imp_expert_reward.mean()

      # In some cases the online reward can go as low as -50
      # even when the mean is positive. To be robust to these outliers, we just
      # clip the negative online rewards, as the role of this term is to
      # regularize the large positive values.
      # This KL estimator is motivated in http://joschu.net/blog/kl-approx.html
      if self.negative_reward:
        online_log_ratio = online_reward + self.reward_scaling * MAX_REWARD
      else:
        online_log_ratio = online_reward
      safe_online_log_ratio = jnp.maximum(online_log_ratio, -5.0)
      # the estimator is log r + 1/r - 1, and the reward is alpha log r
      policy_kl_est = (
          jnp.exp(-safe_online_log_ratio) - 1. + online_log_ratio
      ).mean()

      def non_batch_state_action_value_fn(s, a):
        batch_s = tree.map_structure(lambda f: f[None, ...], s)
        return state_action_value_fn(batch_s, a[None, ...])[0]
      dqda = jax.vmap(
          jax.jacrev(non_batch_state_action_value_fn, argnums=1), in_axes=(0))
      grads = dqda(demonstration_transitions.observation,
                   demonstration_transitions.action)
      grad_norm = jnp.sqrt((grads**2).sum(axis=1).mean())

      loss = sbe
      if self.refine_reward:
        loss -= expert_reward_mean
        loss += self.scale_factor * policy_kl_est
      loss += self.grad_norm_sf * grad_norm

      metrics = {
          "critic_action_grad_norm": grad_norm,
          "sq_bellman_error": sbe,
          "expert_reward": expert_reward.mean(),
          "online_reward": online_reward.mean(),
          "kl_est": policy_kl_est,
      }
      return loss, metrics

    return objective

  def reward_factory(self) -> RewardFact:
    """CSIL's reward function is policy-derived."""

    def reward_factory_(
        state_action_reward: StateActionFunc,
        state_action_value_function: StateActionFunc,
        state_value_function: StochStateFunc,
        discount_factor: float,
    ) -> StateActionNextStateFunc:
      del state_action_value_function, state_value_function, discount_factor

      def reward_fn(
          state: jnp.ndarray,
          action: jnp.ndarray,
          next_state: jnp.ndarray,
          discount: jnp.ndarray,
          value_key: jnp.ndarray,
      ) -> jnp.ndarray:
        del next_state, discount, value_key
        return state_action_reward(state, action)

      return reward_fn

    return reward_factory_


@dataclasses.dataclass
class PretrainingConfig:
  """Parameters for pretraining a model."""

  dataset_factory: DemonstrationFact
  learning_rate: float
  steps: int
  seed: int
  loss: Optional[Losses] = None  # Used for policy pretraining.
  use_as_reference: bool = False  # Use pretrained policy as prior.


def null_data_factory(
    n_demonstrations: int, seed: int
) -> Iterator[types.Transition]:
  del n_demonstrations, seed
  raise NotImplementedError()


SilConfigTypes = Union[InverseSoftQConfig, ProximalPointConfig, CoherentConfig]


@dataclasses.dataclass
class SILConfig(sac_config.SACConfig):
  """Configuration options for soft imitation learning."""

  # Imitation learning hyperparameters.
  expert_demonstration_factory: DemonstrationFact = null_data_factory
  imitation: SilConfigTypes = (
      dataclasses.field(default_factory=InverseSoftQConfig))
  actor_bc_loss: bool = False
  policy_pretraining: Optional[List[PretrainingConfig]] = None
  critic_pretraining: Optional[PretrainingConfig] = None
  actor_learning_rate: float = 3e-4
  reward_learning_rate: float = 3e-4
  critic_learning_rate: float = 3e-4
  critic_actor_update_ratio: int = 1
  alpha_learning_rate: float = 3e-4
  alpha_init: float = 1.0
  damping: float = 0.0  # Entropy constraint damping.
