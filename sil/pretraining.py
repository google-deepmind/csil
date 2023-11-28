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

"""Methods to pretrain networks, e.g. policies, with regression.
"""
import time
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Sequence, Tuple

from acme import specs
from acme import types
from acme.agents.jax import bc
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import counting
from acme.utils import experiment_utils
from acme.utils import loggers
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
import tree

from sil import config

tfd = tfp.distributions

LoggerFactory = Callable[[], loggers.Logger]


BCLossWithoutAux = bc.losses.BCLossWithoutAux
BCLossWithAux = bc.losses.BCLossWithAux
LossArgs = [
    bc.networks.BCNetworks,
    networks_lib.Params,
    networks_lib.PRNGKey,
    types.Transition,
]
Metrics = Dict[str, jnp.ndarray]

ExtendedBCLossWithAux = Tuple[
    Callable[LossArgs, Tuple[jnp.ndarray, Metrics]],
    Callable[[networks_lib.Params], networks_lib.Params],
    Callable[[networks_lib.Params], networks_lib.Params],
]


def no_param_change(params: networks_lib.Params) -> networks_lib.Params:
  return params


def weight_decay(params: networks_lib.Params) -> jnp.ndarray:
  """Used for weight decay loss terms."""
  return  0.5 * sum(
      jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))


def mse(action_dimension: int) -> ExtendedBCLossWithAux:
  """Log probability loss."""
  del action_dimension
  def loss(
      networks: bc.networks.BCNetworks,
      params: networks_lib.Params,
      key: jax_types.PRNGKey,
      transitions: types.Transition,
  ) -> Tuple[jnp.ndarray, Metrics]:
    dist = networks.policy_network.apply(
        params, transitions.observation, is_training=True, key=key,
        train_encoder=True,
    )
    sample = networks.sample_fn(dist, key)
    entropy = -networks.log_prob(dist, sample).mean()
    mean_sq_error = ((dist.mode() - transitions.action) ** 2).mean()
    metrics = {
        "mse": mean_sq_error,
        "nllh": -networks.log_prob(dist, transitions.action).mean(),
        "ent": entropy,
    }
    return mean_sq_error, metrics

  return loss, no_param_change, no_param_change


def faithful_loss(action_dimension: int) -> ExtendedBCLossWithAux:
  """Combines mean-squared error and negative log-likeihood.

  Uses stop-gradients to ensure 'faithful' MSE fit and uncertainty
  quantification.'

  Args:
    action_dimension: (Unused for this loss)

  Returns:
    loss function
    parameter extender: for adding loss terms to pytree params
    parameter retracter: for removing loss terms from pytree params
  """
  del action_dimension
  def loss(
      networks: bc.networks.BCNetworks,
      params: networks_lib.Params,
      key: jax_types.PRNGKey,
      transitions: types.Transition,
  ) -> Tuple[jnp.ndarray, Metrics]:
    dist, cut_dist = networks.policy_network.apply(
        params,
        transitions.observation,
        is_training=True,
        key=key,
        faithful_distributions=True,
        train_encoder=True,
    )
    sample = networks.sample_fn(dist, key)
    entropy = -networks.log_prob(cut_dist, sample).mean()
    mean_sq_error = ((dist.mode() - transitions.action) ** 2).mean()
    nllh = -networks.log_prob(cut_dist, transitions.action).mean()

    loss_ = mean_sq_error + nllh

    metrics = {
        "mse": mean_sq_error,
        "nllh": nllh,
        "ent": entropy,
    }

    return loss_, metrics

  return loss, no_param_change, no_param_change


def negative_loglikelihood(action_dimension: int) -> ExtendedBCLossWithAux:
  """Negative log likelihood loss."""

  del action_dimension
  def loss(
      networks: bc.networks.BCNetworks,
      params: networks_lib.Params,
      key: jax_types.PRNGKey,
      transitions: types.Transition,
  ) -> Tuple[jnp.ndarray, Metrics]:
    dist = networks.policy_network.apply(
        params, transitions.observation, is_training=True, key=key,
        train_encoder=True,
    )
    nllh = -networks.log_prob(dist, transitions.action).mean()
    sample = networks.sample_fn(dist, key)
    entropy = -networks.log_prob(dist, sample).mean()
    metrics = {
        "mse": ((dist.mode() - transitions.action) ** 2).mean(),
        "nllh": nllh,
        "ent": entropy,
    }
    return nllh, metrics

  return loss, no_param_change, no_param_change


def zero_debiased_faithful_loss(
    action_dimension: int,
) -> ExtendedBCLossWithAux:
  """Combines mean-squared error and negative log-likeihood.

  Uses stop-gradients to ensure 'faithful' MSE fit and uncertainty
  quantification.'

  Args:
    action_dimension: (Unused for this loss)

  Returns:
    loss function
    parameter extender: for adding loss terms to pytree params
    parameter retracter: for removing loss terms from pytree params
  """

  def loss(
      networks: bc.networks.BCNetworks,
      params: networks_lib.Params,
      key: jax_types.PRNGKey,
      transitions: types.Transition,
  ) -> Tuple[jnp.ndarray, Metrics]:
    policy_params = params["model"]
    dist, cut_dist = networks.policy_network.apply(
        policy_params,
        transitions.observation,
        is_training=True,
        key=key,
        faithful_distributions=True,
        train_encoder=True,
    )
    sample = networks.sample_fn(dist, key)
    entropy = -networks.log_prob(cut_dist, sample).mean()
    mean_sq_error = ((dist.mode() - transitions.action) ** 2).mean(axis=1)
    nllh = -networks.log_prob(cut_dist, transitions.action)

    mse1 = jnp.expand_dims(mean_sq_error, axis=1)
    nllh1 = jnp.expand_dims(nllh, axis=1)

    loss_params = params["loss"]
    w_virtual = loss_params["w_virtual"]
    w = jax.nn.softmax(w_virtual)
    log_w = jnp.expand_dims(jnp.log(w), axis=0)
    iso_sigma = jax.nn.softplus(loss_params["sigma_virtual"])
    sigma = iso_sigma * jnp.ones((action_dimension,))

    dist0 = tfd.MultivariateNormalDiag(
        loc=jnp.zeros((action_dimension,)), scale_diag=sigma
    )

    mse0 = jnp.expand_dims((transitions.action**2).mean(axis=1), axis=1)
    mse0 = mse0 / iso_sigma
    nllh0 = -jnp.expand_dims(dist0.log_prob(transitions.action), axis=1)

    mse_ = jnp.concatenate([mse0, mse1], axis=1)
    nllh_ = jnp.concatenate([nllh0, nllh1], axis=1)

    # Do a mixture likelihood.
    # log sum_i w_i p(x|i) = logsumexp(log_p(x|i) + log_w) along a new dimension
    # MSE needs temp for sharp minimum estimate.
    mse_lse_temp = 1000.0
    # Don't do logsumexp do the other form.
    mmse = (
        -jax.scipy.special.logsumexp((mse_lse_temp * -mse_ + log_w), axis=1)
        / mse_lse_temp
    )
    mnllh = -jax.scipy.special.logsumexp((-nllh_ + log_w), axis=1)
    mmse, mnllh = mmse.mean(), mnllh.mean()

    loss_ = mmse + mnllh
    metrics = {
        "mse": mmse,
        "nllh": mnllh,
        "ent": entropy,
    }

    return loss_, metrics

  def extend_params(params: networks_lib.Params) -> networks_lib.Params:
    new_params = {
        "loss": {
            "w_virtual": jnp.array([0.0, 1.0]),
            "sigma_virtual": -1.0 * jnp.ones((1,)),
        },
        "model": params,
    }
    return new_params

  def retract_params(params: networks_lib.Params) -> networks_lib.Params:
    return params["model"] if "model" in params else params

  return loss, extend_params, retract_params


# Map enum to loss function.
_LOOKUP = {
    config.Losses.FAITHFUL: faithful_loss,
    config.Losses.DBFAITHFUL: zero_debiased_faithful_loss,
    config.Losses.MSE: mse,
    config.Losses.NLLH: negative_loglikelihood,
}


def get_loss_function(
    loss_type: config.Losses,
) -> Callable[[int], ExtendedBCLossWithAux]:
  assert loss_type in _LOOKUP
  return _LOOKUP[loss_type]


TerminateCondition = Callable[[list[dict[str, jnp.ndarray]]], bool]


class EarlyStoppingBCLearner(bc.BCLearner):
  """Behavioural cloning learner that stops based on metrics."""

  def __init__(self, terminate_condition: TerminateCondition, *args, **kwargs):
    self.metrics = []
    self.terminate_condition = terminate_condition
    self.terminate = False
    super().__init__(*args, **kwargs)

  def step(self):
    # Get a batch of Transitions.
    transitions = next(self._prefetching_iterator)
    self._state, metrics = self._sgd_step(self._state, transitions)
    metrics = utils.get_from_first_device(metrics)
    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    # Increment counts and record the current time.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})
    self.metrics += [metrics,]
    self.terminate = self.terminate_condition(self.metrics)


def behavioural_cloning_pretraining(
    seed: int,
    env_spec: specs.EnvironmentSpec,
    dataset_factory: Callable[[int], Iterator[types.Transition]],
    policy: networks_lib.FeedForwardNetwork,
    loss: config.Losses = config.Losses.FAITHFUL,
    num_steps: int = 40_000,
    learning_rate: float = 1e-4,
    logger: Optional[loggers.Logger] = None,
    name: str = "",
) -> networks_lib.Params:
  """Trains the policy and returns the params single-threaded training loop.

  Args:
    seed: Random seed for training.
    env_spec: Environment specification.
    dataset_factory: A function that returns an iterator with demonstrations to
      be imitated.
    policy: Policy network model.
    loss: loss type for pretraining (e.g. MSE, log likelihood, ...)
    num_steps: Number of training steps.
    learning_rate: Used for regression.
    logger: Optional external object for logging.
    name: Name used for logger.

  Returns:
    The trained network params.
  """
  key = random.PRNGKey(seed)

  logger = logger or experiment_utils.make_experiment_logger(f"pretrainer_policy{name}")

  # Train using log likelihood.
  n_actions = np.prod(env_spec.actions.shape)
  loss_fn = _LOOKUP[loss]
  bc_loss, extend_params, retract_params = loss_fn(n_actions)

  bc_policy_network = bc.convert_to_bc_network(policy)
  # Add loss terms to params here.
  policy_network = bc.BCPolicyNetwork(
      lambda key: extend_params(policy.init(key)), bc_policy_network.apply
  )

  bc_network = bc.BCNetworks(
      policy_network=policy_network,
      log_prob=lambda params, acts: params.log_prob(acts),
      # For BC agent, the sample_fn is used for evaluation.
      sample_fn=lambda params, key: params.sample(seed=key),
  )

  dataset = dataset_factory(seed)

  counter = counting.Counter(prefix="policy_pretrainer", time_delta=0.0)

  history = 50
  ent_threshold = -2 * n_actions

  def terminate_condition(metrics: Sequence[Dict[str, jnp.ndarray]]) -> bool:
    if len(metrics) < history:
      return False
    else:
      return all(m["ent"] < ent_threshold for m in metrics[-history:])

  learner = EarlyStoppingBCLearner(
      terminate_condition=terminate_condition,
      loss_fn=bc_loss,
      optimizer=optax.adam(learning_rate=learning_rate),
      random_key=key,
      networks=bc_network,
      prefetching_iterator=utils.sharded_prefetch(dataset),
      loss_has_aux=True,
      num_sgd_steps_per_step=1,
      logger=logger,
      counter=counter,)

  # Train the agent.
  for _ in range(num_steps):
    learner.step()
    # learner.terminate is available

  policy_and_loss_params = learner.get_variables(["policy"])[0]
  del learner  # Ensure logger is closed.
  # Remove loss terms from params here.
  return retract_params(policy_and_loss_params)


class TrainingState(NamedTuple):
  params: networks_lib.Params
  target_params: networks_lib.Params
  opt_state: optax.OptState


def critic_pretraining(
    seed: int,
    dataset_factory: Callable[[int], Iterator[types.Transition]],
    critic: networks_lib.FeedForwardNetwork,
    critic_params: networks_lib.Params,
    reward: networks_lib.FeedForwardNetwork,
    reward_params: networks_lib.Params,
    discount_factor: float,
    num_steps: int = 10_000,
    learning_rate: float = 5e-3,
    counter: Optional[counting.Counter] = None,
    logger: Optional[loggers.Logger] = None,
) -> networks_lib.Params:
  """Pretrain the critic using a SARSA loss.

  Args:
   seed: for randomized training
   dataset_factory: SARSA data iterator for pretraining
   critic: critic function
   critic_params: initial critic params
   reward: reward function
   reward_params: known reward params
   discount_factor: discount used for Bellman equation
   num_steps: number of update steps
   learning_rate: learning rate of optimizer
   counter: used for logging
   logger: Optional external object for logging.

  Returns:
    Trained critic params.
  """
  key = jax.random.PRNGKey(seed)
  optimiser = optax.adam(learning_rate)

  initial_opt_state = optimiser.init(critic_params)

  state = TrainingState(critic_params, critic_params, initial_opt_state)

  dataset_iterator = dataset_factory(seed)

  tau = 0.005

  sample = next(dataset_iterator)
  assert "next_action" in sample.extras, "Require SARSA dataset."

  @jax.jit
  def loss(
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      observation: jnp.ndarray,
      action: jnp.ndarray,
      next_observation: jnp.ndarray,
      next_action: jnp.ndarray,
      discount: jnp.ndarray,
      key: jax_types.PRNGKey,
  ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """."""
    del key

    r = jnp.ravel(reward.apply(reward_params, observation, action))

    next_v_target = critic.apply(
        target_params, next_observation, next_action).min(axis=-1)
    next_v_target = jax.lax.stop_gradient(next_v_target)
    discount_ = discount_factor * discount
    q_sarsa_target = jnp.expand_dims(r +  discount_ * next_v_target, -1)
    q = critic.apply(params, observation, action)
    sarsa_loss = ((q_sarsa_target - q) ** 2).mean()

    def nonbatch_critic(s, a):
      batch_s = tree.map_structure(lambda f: f[None, ...], s)
      return critic.apply(params, batch_s, a[None, ...])[0]

    dqda = jax.vmap(jax.jacfwd(nonbatch_critic, argnums=1), in_axes=(0))
    grads = dqda(observation, action)
    # Sum over actions, average over the rest.
    grad_norm = jnp.sqrt((grads**2).sum(axis=-1).mean())

    loss = sarsa_loss + grad_norm

    metrics = {
        "loss": loss,
        "sarsa_loss": sarsa_loss,
        "grad_norm": grad_norm,
    }

    return loss, metrics

  @jax.jit
  def step(transition, state, key):
    if "next_action" in transition.extras:
      next_action = transition.extras["next_action"]
    else:
      next_action = transition.action
    values, grads = jax.value_and_grad(loss, has_aux=True)(
        state.params,
        state.target_params,
        transition.observation,
        transition.action,
        transition.next_observation,
        next_action,
        transition.discount,
        key,
    )
    _, metrics = values
    updates, opt_state = optimiser.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    target_params = jax.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, state.target_params, params
    )
    return TrainingState(params, target_params, opt_state), metrics

  timestamp = time.time()
  counter = counter or counting.Counter(
      prefix="pretrainer_critic", time_delta=0.0
  )
  logger = logger or loggers.make_default_logger(
      "pretrainer_critic",
      asynchronous=False,
      serialize_fn=utils.fetch_devicearray,
      steps_key=counter.get_steps_key(),
  )
  for i in range(num_steps):
    _, key = jax.random.split(key)
    transitions = next(dataset_iterator)
    state, metrics = step(transitions, state, key)
    metrics["step"] = i
    timestamp_ = time.time()
    elapsed_time = timestamp_ - timestamp
    timestamp = timestamp_
    counts = counter.increment(steps=1, walltime=elapsed_time)
    logger.write({**metrics, **counts})

  logger.close()

  return state.params
