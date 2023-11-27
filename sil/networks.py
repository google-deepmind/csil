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

"""Soft imitation networks definition.

Builds heavily on acme/agents/jax/sac/networks.py
(at https://github.com/google-deepmind/acme)
"""

import dataclasses
import enum
import math
from typing import Any, Optional, Tuple, Callable, Sequence, Union

from acme import core
from acme import specs
from acme.agents.jax.sac import networks as sac_networks
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.jax import utils
import haiku as hk
import haiku.initializers as hk_init
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from sil import config as sil_config

tfd = tfp.distributions
tfb = tfp.bijectors


class PolicyArchitectures(enum.Enum):
  """Variations of policy architectures used in SIL methods."""
  MLP = 'mlp'
  MIXMLP = 'mixmlp'
  HETSTATSIN = 'hetstatsin'
  HETSTATTRI = 'hetstattri'
  HETSTATPRELU = 'hetstatprelu'
  MIXHETSTATSIN = 'mixhetstatsin'
  MIXHETSTATTRI = 'mixhetstattri'
  MIXHETSTATPRELU = 'mixhetstatprelu'

  def __str__(self) -> str:
    return self.value


class CriticArchitectures(enum.Enum):
  MLP = 'mlp'
  DOUBLE_MLP = 'double_mlp'   # Used for SAC-based methods that use two critics.
  LNMLP = 'lnmlp'
  DOUBLE_LNMLP = 'double_lnmlp'
  STATSIN = 'statsin'
  STATTRI = 'stattri'
  STATPRELU = 'statprelu'

  def __str__(self) -> str:
    return self.value


class RewardArchitectures(enum.Enum):
  MLP = 'mlp'
  LNMLP = 'lnmlp'
  PCSIL = 'pos_csil'
  NCSIL = 'neg_csil'
  PCONST = 'pos_const'
  NCONST = 'neg_const'

  def __str__(self) -> str:
    return self.value


def observation_encoder(inputs: Any) -> jnp.ndarray:
  """Function that transforms observations into the correct vectors."""
  if isinstance(inputs, jnp.ndarray):
    return inputs
  else:
    raise ValueError(f'Cannot convert type {type(inputs)}.')


def update_encoder(params: networks_lib.Params,
                   reference: networks_lib.Params) -> networks_lib.Params:
  predicate = lambda module_name, name, value: 'encoder' in module_name
  _, params_head = hk.data_structures.partition(predicate, params)
  ref_enc, _ = hk.data_structures.partition(predicate, reference)
  return hk.data_structures.merge(params_head, ref_enc)


class Sequential(hk.Module):
  """Sequentially calls the given list of layers."""

  def __init__(
      self,
      layers: Sequence[Callable[..., Any]],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.layers = tuple(layers)

  def __call__(self, inputs, *args, **kwargs):
    """Calls all layers sequentially."""
    out = inputs
    last_idx = len(self.layers) - 1
    for i, layer in enumerate(self.layers):
      if i == last_idx:
        out = layer(out, *args, **kwargs)
      else:
        out = layer(out)
    return out


@dataclasses.dataclass
class SILNetworks:
  """Network and pure functions for the soft imitation agent."""

  environment_specs: specs.EnvironmentSpec
  policy_architecture: PolicyArchitectures
  bc_policy_architecture: PolicyArchitectures
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork
  reward_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  log_prob_prior: Callable[[jnp.ndarray], jnp.ndarray]
  sample: networks_lib.SampleFn
  bc_policy_network: networks_lib.FeedForwardNetwork
  reward_policy_coherence: bool = False
  sample_eval: Optional[networks_lib.SampleFn] = None

  def to_sac(self, using_bc_policy: bool = False) -> sac_networks.SACNetworks:
    """Cast to SAC policy to make use of the SAC helpers."""
    policy_network = (
        self.bc_policy_network if using_bc_policy else self.policy_network
    )
    return sac_networks.SACNetworks(
        policy_network,
        self.critic_network,
        self.log_prob,
        self.sample,
        self.sample_eval,
    )


#  From acme/agents/jax/cql/networks.py
def apply_and_sample_n(
    key: networks_lib.PRNGKey,
    networks: SILNetworks,
    params: networks_lib.Params,
    obs: jnp.ndarray,
    num_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Applies the policy and samples num_samples actions."""
  dist_params = networks.policy_network.apply(params, obs)
  sampled_actions = jnp.array(
      [
          networks.sample(dist_params, key_n)
          for key_n in jax.random.split(key, num_samples)
      ]
  )
  sampled_log_probs = networks.log_prob(dist_params, sampled_actions)
  return sampled_actions, sampled_log_probs


def default_models_to_snapshot(
    networks: SILNetworks, spec: specs.EnvironmentSpec
):
  """Defines default models to be snapshotted."""
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.zeros_like(spec.actions)
  dummy_key = jax.random.PRNGKey(0)

  def critic_network(source: core.VariableSource) -> types.ModelToSnapshot:
    params = source.get_variables(['critic'])[0]
    return types.ModelToSnapshot(
        networks.critic_network.apply,
        params,
        {'obs': dummy_obs, 'action': dummy_action},
    )

  def reward_network(source: core.VariableSource) -> types.ModelToSnapshot:
    params = source.get_variables(['reward'])[0]
    return types.ModelToSnapshot(
        networks.critic_network.apply,
        params,
        {'obs': dummy_obs, 'action': dummy_action},
    )

  def default_training_actor(
      source: core.VariableSource) -> types.ModelToSnapshot:
    params = source.get_variables(['policy'])[0]
    return types.ModelToSnapshot(
        sac_networks.apply_policy_and_sample(
            networks.to_sac(), eval_mode=False
        ),
        params,
        {'key': dummy_key, 'obs': dummy_obs},
    )

  def default_eval_actor(
      source: core.VariableSource) -> types.ModelToSnapshot:
    params = source.get_variables(['policy'])[0]
    return types.ModelToSnapshot(
        sac_networks.apply_policy_and_sample(networks.to_sac(), eval_mode=True),
        params,
        {'key': dummy_key, 'obs': dummy_obs},
    )

  return {
      'critic_network': critic_network,
      'reward_network': reward_network,
      'default_training_actor': default_training_actor,
      'default_eval_actor': default_eval_actor,
  }


default_init_normal = hk.initializers.VarianceScaling(
    0.333, 'fan_out', 'normal'
)
# If 1D regression plotting, 0.2 is more sensible.
# It's crucial this is not changed, it doesn't work otherwise.
# Acme SAC uses this
default_init_uniform = hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')


class ClampedScaleNormalTanhDistribution(hk.Module):
  """Module that produces a variance-clampled TanhTransformedDistribution."""

  def __init__(
      self,
      num_dimensions: int,
      max_log_scale: float = 0.0,
      min_log_scale: float = -5.0,
      w_init: hk_init.Initializer = hk_init.Orthogonal(),
      b_init: hk_init.Initializer = hk_init.Constant(0.0),
      name: str = 'ClampedScaleNormalTanhDistribution',
  ):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of a distribution.
      max_log_scale: Maximum log standard deviation.
      min_log_scale: Minimum log standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
      name: name of model that is passed to the parameters
    """
    super().__init__(name=name)
    assert max_log_scale > min_log_scale
    self._min_log_scale = min_log_scale
    self._log_scale_range = max_log_scale - self._min_log_scale
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

  def __call__(
      self, inputs: jnp.ndarray, faithful_distributions: bool = False
  ) -> Union[tfd.Distribution, Tuple[tfd.Distribution, tfd.Distribution]]:
    loc = self._loc_layer(inputs)
    if faithful_distributions:
      inputs_ = jax.lax.stop_gradient(inputs)
    else:
      inputs_ = inputs

    log_scale_raw = self._scale_layer(inputs_)  # Operating range around [-1, 1]
    log_scale_norm = 0.5 * (jax.nn.tanh(log_scale_raw) + 1.0)  # Range [0, 1]
    scale = self._min_log_scale + self._log_scale_range * log_scale_norm
    scale = math.sqrt(sil_config.MIN_VAR) + jnp.exp(scale)
    distribution = tfd.Normal(loc=loc, scale=scale)
    transformed_dist = tfd.Independent(
        networks_lib.TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1,
    )
    if faithful_distributions:
      cut_dist = tfd.Normal(loc=jax.lax.stop_gradient(loc), scale=scale)
      cut_transformed_dist = tfd.Independent(
          networks_lib.TanhTransformedDistribution(cut_dist),
          reinterpreted_batch_ndims=1,
      )
      return transformed_dist, cut_transformed_dist
    else:
      return transformed_dist


class MixtureClampedScaleNormalTanhDistribution(
    ClampedScaleNormalTanhDistribution):
  """Module that produces a variance-clampled TanhTransformedDistribution."""

  def __init__(
      self,
      num_dimensions: int,
      n_mixture: int,
      max_log_scale: float = 0.0,
      min_log_scale: float = -5.0,
      w_init: hk_init.Initializer = hk_init.Orthogonal(),
      b_init: hk_init.Initializer = hk_init.Constant(0.0),
  ):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of a distribution.
      n_mixture: numver of mixture components.
      max_log_scale: Maximum log standard deviation.
      min_log_scale: Minimum log standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    self.n_mixture = n_mixture
    self.n_outputs = num_dimensions
    super().__init__(
        num_dimensions=num_dimensions * n_mixture,
        max_log_scale=max_log_scale,
        min_log_scale=min_log_scale,
        w_init=w_init,
        b_init=b_init,
        name='MixtureClampedScaleNormalTanhDistribution',
    )

  def __call__(
      self, inputs: jnp.ndarray, faithful_distributions: bool = False
  ) -> Union[tfd.Distribution, Tuple[tfd.Distribution, tfd.Distribution]]:
    loc = self._loc_layer(inputs)
    if faithful_distributions:
      inputs_ = jax.lax.stop_gradient(inputs)
    else:
      inputs_ = inputs

    log_scale_raw = self._scale_layer(inputs_)  # Operating range around [-1, 1]
    log_scale_norm = 0.5 * (jax.nn.tanh(log_scale_raw) + 1.0)  # range [0, 1]
    scale = self._min_log_scale + self._log_scale_range * log_scale_norm
    scale = math.sqrt(sil_config.MIN_VAR) + jnp.exp(scale)

    log_mixture_weights = hk.get_parameter(
        'log_mixture_weights',
        [self.n_mixture],
        init=hk.initializers.Constant(1.0),
    )
    mixture_weights = jax.nn.softmax(log_mixture_weights)
    mixture_distribution = tfd.Categorical(probs=mixture_weights)

    def make_mixture(location, scale, weights):
      distribution = tfd.Normal(loc=location, scale=scale)

      transformed_distribution = tfd.Independent(
          networks_lib.TanhTransformedDistribution(distribution),
          reinterpreted_batch_ndims=1,
      )

      return MixtureSameFamily(
          mixture_distribution=weights,
          components_distribution=transformed_distribution,
      )

    mean = loc.reshape((-1, self.n_mixture, self.n_outputs))
    stddev = scale.reshape((-1, self.n_mixture, self.n_outputs))
    mixture = make_mixture(mean, stddev, mixture_distribution)
    if faithful_distributions:
      cut_mixture = make_mixture(jax.lax.stop_gradient(mean),
                                 stddev, mixture_distribution)
      return mixture, cut_mixture
    else:
      return mixture


def _triangle_activation(x: jnp.ndarray) -> jnp.ndarray:
  z = jnp.floor(x / jnp.pi + 0.5)
  return (x - jnp.pi * z) * (-1) ** z


@jax.jit
def triangle_activation(x: jnp.ndarray) -> jnp.ndarray:
  pdiv2sqrt2 = 1.1107207345
  return pdiv2sqrt2 * _triangle_activation(x)


@jax.jit
def periodic_relu_activation(x: jnp.ndarray) -> jnp.ndarray:
  pdiv4 = 0.785398163
  pdiv2 = 1.570796326
  return (_triangle_activation(x) + _triangle_activation(x + pdiv2)) * pdiv4


@jax.jit
def sin_cos_activation(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.sin(x) + jnp.cos(x)


@jax.jit
def hard_sin(x: jnp.ndarray) -> jnp.ndarray:
  pdiv4 = 0.785398163  # π/4
  return periodic_relu_activation(x - pdiv4)


@jax.jit
def hard_cos(x: jnp.ndarray) -> jnp.ndarray:
  pdiv4 = 0.785398163  # π/4
  return periodic_relu_activation(x + pdiv4)


gaussian_init = hk.initializers.RandomNormal(1.0)


class StationaryFeatures(hk.Module):
  """Stationary feature layer.

  Combines an MLP feature component (with bottleneck output) into a relatively
  wider feature layer that has periodic activation function. The from of the
  final weight distribution and periodic activation dictates the nature of the
  parametic stationary process.

  For more details see
  Periodic Activation Functions Induce Stationarity, Meronen et al.
  https://arxiv.org/abs/2110.13572
  """

  def __init__(
      self,
      num_dimensions: int,
      layers: Sequence[int],
      feature_dimension: int = 512,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
      stationary_activation: Callable[
          [jnp.ndarray], jnp.ndarray
      ] = sin_cos_activation,
      stationary_init: hk.initializers.Initializer = gaussian_init,
      layer_norm_mlp: bool = False,
  ):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of the output distribution.
      layers: feature MLP architecture up to the feature layer.
      feature_dimension: size of feature layer.
      activation: Activation of MLP network.
      stationary_activation: Periodic activation of feature layer.
      stationary_init: Random initialization of last layer
      layer_norm_mlp: Use layer norm in the first MLP layer.
    """
    super().__init__(name='StationaryFeatures')
    self.output_dimension = num_dimensions
    self.feature_dimension = feature_dimension
    self.stationary_activation = stationary_activation
    self.stationary_init = stationary_init
    if layer_norm_mlp:
      self.mlp = networks_lib.LayerNormMLP(
          list(layers),
          activation=activation,
          w_init=hk.initializers.Orthogonal(),
          activate_final=True,
      )
    else:
      self.mlp = hk.nets.MLP(
          list(layers),
          activation=activation,
          w_init=hk.initializers.Orthogonal(),
          activate_final=True,
      )

  def features(self, inputs: jnp.ndarray) -> jnp.ndarray:
    input_dimension = inputs.shape[-1]

    # While the theory says that these random weights should be fixed, it's
    # crucial in practice to let them be trained. The distribution does not
    # actually change much, so they still contribute to the stationary
    # behaviour, and letting them be trained alleviates potential underfitting.
    random_weights = hk.get_parameter(
        'random_weights',
        [input_dimension, self.feature_dimension // 2],
        init=self.stationary_init,
    )

    log_lengthscales = hk.get_parameter(
        'log_lengthscales',
        [input_dimension],
        init=hk.initializers.Constant(-5.0),
    )

    ls = jnp.diag(jnp.exp(log_lengthscales))
    wx = inputs @ ls @ random_weights
    pdiv4 = 0.785398163  # π/4
    f = jnp.concatenate(
        (
            self.stationary_activation(wx + pdiv4),
            self.stationary_activation(wx - pdiv4),
        ),
        axis=-1,
    )
    return f / math.sqrt(self.feature_dimension)


class StationaryHeteroskedasticNormalTanhDistribution(StationaryFeatures):
  """Module that produces a stationary TanhTransformedDistribution."""

  def __init__(
      self,
      num_dimensions: int,
      layers: Sequence[int],
      feature_dimension: int = 512,
      prior_variance: float = 1.0,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
      stationary_activation: Callable[
          [jnp.ndarray], jnp.ndarray
      ] = sin_cos_activation,
      stationary_init: hk.initializers.Initializer = gaussian_init,
      layer_norm_mlp: bool = False,
  ):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of the output distribution.
      layers: feature MLP architecture up to the feature layer.
      feature_dimension: size of feature layer.
      prior_variance: initial variance of the predictive.
      activation: Activation of MLP network.
      stationary_activation: Periodic activation of feature layer.
      stationary_init: Random initialization of last layer
      layer_norm_mlp: Use layer norm in the first MLP layer.
    """
    self.prior_var = prior_variance
    self.prior_stddev = np.sqrt(prior_variance)
    super().__init__(
        num_dimensions,
        layers,
        feature_dimension,
        activation,
        stationary_activation,
        stationary_init,
        layer_norm_mlp
    )

  def __call__(
      self, inputs: jnp.ndarray, faithful_distributions: bool = False
  ) -> Union[tfd.Distribution, Tuple[tfd.Distribution, tfd.Distribution]]:
    inputs = self.mlp(inputs)
    features = self.features(inputs)
    if faithful_distributions:
      features_ = jax.lax.stop_gradient(features)
    else:
      features_ = features

    loc_weights = hk.get_parameter(
        'loc_weights',
        [self.feature_dimension, self.output_dimension],
        init=hk.initializers.Constant(0.0),
    )

    # Parameterize the PSD matrix in lower triangular form as a 'raw' vector.
    # This minimizes the memory footprint to between 50-75% of the full matrix.
    n_sqrt = self.feature_dimension * (self.feature_dimension + 1) // 2
    scale_cross_weights_sqrt_raw = hk.get_parameter(
        'scale_cross_weights_sqrt',
        [self.output_dimension, n_sqrt],
        init=hk.initializers.Constant(0.0),
    )
    # convert vector into a lower triagular matrix with exponentiated diagonal,
    # so a vector of zeros becomes the identity matrix.
    b = tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)
    scale_cross_weights_sqrt = jax.vmap(b.forward)(scale_cross_weights_sqrt_raw)
    loc = features @ loc_weights
    # Cholesky decompositon: A = LL^T where L is lower triangular
    # Variance is diagonal of x @ A @ x.T = x @ L @ L.T @ x.T
    # so first compute x @ L per output d
    var_sqrt = jnp.einsum('dij,bi->bdj', scale_cross_weights_sqrt, features_)
    var = jnp.einsum('bdi,bdi->bd', var_sqrt, var_sqrt)

    scale = self.prior_stddev * jnp.tanh(jnp.sqrt(sil_config.MIN_VAR + var))

    distribution = tfd.Normal(loc=loc, scale=scale)

    transformed_distribution = tfd.Independent(
        networks_lib.TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1,
    )

    if faithful_distributions:
      cut_distribution = tfd.Normal(loc=jax.lax.stop_gradient(loc), scale=scale)
      cut_transformed_distribution = tfd.Independent(
          networks_lib.TanhTransformedDistribution(cut_distribution),
          reinterpreted_batch_ndims=1,
      )
      return transformed_distribution, cut_transformed_distribution
    else:
      return transformed_distribution


class MixtureSameFamily(tfd.MixtureSameFamily):
  """MixtureSameFamily with mode computation."""

  def mode(self) -> jnp.ndarray:
    """Return the mode of the modal mixture distribution."""
    mode_model = self.mixture_distribution.mode()
    modes = self.components_distribution.mode()
    return modes[:, mode_model, :]


class MixtureStationaryHeteroskedasticNormalTanhDistribution(
    StationaryHeteroskedasticNormalTanhDistribution
):
  """Module that produces a stationary TanhTransformedDistribution."""

  def __init__(
      self,
      num_dimensions: int,
      n_mixture: int,
      layers: Sequence[int],
      feature_dimension: int = 512,
      prior_variance: float = 1.0,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
      stationary_activation: Callable[
          [jnp.ndarray], jnp.ndarray
      ] = sin_cos_activation,
      stationary_init: hk.initializers.Initializer = gaussian_init,
      layer_norm_mlp: bool = False,
  ):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of the output distribution.
      n_mixture: Number of mixture components.
      layers: feature MLP architecture up to the feature layer.
      feature_dimension: size of feature layer.
      prior_variance: initial variance of the predictive.
      activation: Activation of MLP network.
      stationary_activation: Periodic activation of feature layer.
      stationary_init: Random initialization of last layer
      layer_norm_mlp: Use layer norm in the first MLP layer.
    """
    self.n_mixture = n_mixture
    super().__init__(
        num_dimensions=num_dimensions,
        layers=layers,
        feature_dimension=feature_dimension,
        prior_variance=prior_variance,
        activation=activation,
        stationary_activation=stationary_activation,
        stationary_init=stationary_init,
        layer_norm_mlp=layer_norm_mlp,
    )

  def __call__(
      self, inputs: jnp.ndarray, faithful_distributions: bool = False
  ) -> Union[tfd.Distribution, Tuple[tfd.Distribution, tfd.Distribution]]:
    inputs = self.mlp(inputs)
    features = self.features(inputs)
    if faithful_distributions:
      features_ = jax.lax.stop_gradient(features)
    else:
      features_ = features

    loc_weights = hk.get_parameter(
        'loc_weights',
        [self.feature_dimension, self.n_mixture, self.output_dimension],
        init=hk.initializers.Orthogonal(),  # For mixture diversity.
    )

    scale_cross_weights_sqrt = hk.get_parameter(
        'scale_cross_weights_sqrt',
        [self.output_dimension, self.feature_dimension, self.feature_dimension],
        init=hk.initializers.Identity(gain=1.0),
    )

    # batch x n_mixture x d_out
    mean = jnp.einsum('bi,ijk->bjk', features, loc_weights)
    scale_cross_weights_sqrt = jnp.tril(scale_cross_weights_sqrt)
    # Cholesky decompositon: A = LL^T where L is lower triangular
    # Variance is diagonal of x @ A @ x.T = x @ L @ L.T @ x.T
    # so first compute x @ L per output d
    var_sqrt = jnp.einsum('dij,bi->bdj', scale_cross_weights_sqrt, features_)
    var = jnp.einsum('bdi,bdi->bd', var_sqrt, var_sqrt)

    stddev_ = self.prior_stddev * jnp.tanh(jnp.sqrt(sil_config.MIN_VAR + var))
    stddev = jnp.repeat(jnp.expand_dims(stddev_, 1), self.n_mixture, axis=1)

    assert mean.shape == stddev.shape, f'{mean.shape} != {stddev.shape}'

    log_mixture_weights = hk.get_parameter(
        'log_mixture_weights',
        [self.n_mixture],
        init=hk.initializers.Constant(1.0),
    )
    mixture_weights = jax.nn.softmax(log_mixture_weights)
    mixture_distribution = tfd.Categorical(probs=mixture_weights)

    def make_mixture(location, scale, weights):
      distribution = tfd.Normal(loc=location, scale=scale)

      transformed_distribution = tfd.Independent(
          networks_lib.TanhTransformedDistribution(distribution),
          reinterpreted_batch_ndims=1,
      )

      return MixtureSameFamily(
          mixture_distribution=weights,
          components_distribution=transformed_distribution,
      )

    mixture = make_mixture(mean, stddev, mixture_distribution)
    if faithful_distributions:
      cut_mixture = make_mixture(
          jax.lax.stop_gradient(mean), stddev, mixture_distribution)
      return mixture, cut_mixture
    else:
      return mixture


class StationaryMLP(StationaryFeatures):
  """MLP that behaves like the mean function of a stationary process."""

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    inputs = self.mlp(inputs)
    features = self.features(inputs)

    loc_weights = hk.get_parameter(
        'loc_weights',
        [self.feature_dimension, self.output_dimension],
        init=hk.initializers.Constant(0.0),
    )
    return features @ loc_weights


class LayerNormMLP(hk.Module):
  """Simple feedforward MLP torso with initial layer-norm."""

  def __init__(
      self,
      layer_sizes: Sequence[int],
      w_init: hk.initializers.Initializer,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
      activate_final: bool = False,
      name: str = 'feedforward_mlp_torso',
  ):
    """Construct the MLP."""
    super().__init__(name=name)
    assert len(layer_sizes) > 1
    self._network = hk.Sequential([
        hk.Linear(layer_sizes[0], w_init=w_init),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        hk.nets.MLP(
            layer_sizes[1:],
            w_init=w_init,
            activation=activation,
            activate_final=activate_final,
        ),
    ])

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Forwards the policy network."""
    return self._network(inputs)


def prior_policy_log_likelihood(
    env_spec: specs.EnvironmentSpec, policy_architecture: PolicyArchitectures
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """We assume a uniform hyper prior in a [-1, 1] action space."""
  del policy_architecture
  act_spec = env_spec.actions
  num_actions = np.prod(act_spec.shape, dtype=int)

  prior_llh = lambda x: -num_actions * jnp.log(2.0)
  return jax.vmap(prior_llh)


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_architecture: PolicyArchitectures = PolicyArchitectures.MLP,
    critic_architecture: CriticArchitectures = CriticArchitectures.LNMLP,
    reward_architecture: RewardArchitectures = RewardArchitectures.LNMLP,
    policy_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    critic_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    reward_hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    reward_policy_coherence_alpha: Optional[float] = None,
    bc_policy_architecture: Optional[PolicyArchitectures] = None,
    bc_policy_hidden_layer_sizes: Optional[Tuple[int, ...]] = None,
    layer_norm_policy: bool = False,
) -> SILNetworks:
  """Creates networks used by the agent."""

  num_actions = np.prod(spec.actions.shape, dtype=int)

  def _make_actor_fn(
      policy_arch: PolicyArchitectures, hidden_layer_size: Tuple[int, ...]
  ):
    assert len(hidden_layer_size) > 1

    def _actor_fn(obs, *args, train_encoder=False, **kwargs):
      if policy_arch == PolicyArchitectures.MLP:
        mlp = networks_lib.LayerNormMLP if layer_norm_policy else hk.nets.MLP
        network = Sequential([
            mlp(
                list(hidden_layer_size),
                w_init=hk.initializers.Orthogonal(),
                activation=jax.nn.elu,
                activate_final=True,
            ),
            ClampedScaleNormalTanhDistribution(num_actions),
        ])
      elif policy_arch == PolicyArchitectures.MIXMLP:
        mlp = networks_lib.LayerNormMLP if layer_norm_policy else hk.nets.MLP
        network = Sequential([
            mlp(
                list(hidden_layer_size),
                w_init=hk.initializers.Orthogonal(),
                activation=jax.nn.elu,
                activate_final=True,
            ),
            MixtureClampedScaleNormalTanhDistribution(num_actions, n_mixture=5),
        ])
      elif policy_arch == PolicyArchitectures.HETSTATSIN:
        network = StationaryHeteroskedasticNormalTanhDistribution(
            num_actions,
            hidden_layer_size[:-1],
            feature_dimension=hidden_layer_size[-1],
            prior_variance=0.75,
            stationary_activation=sin_cos_activation,
            layer_norm_mlp=layer_norm_policy,
        )
      elif policy_arch == PolicyArchitectures.HETSTATTRI:
        network = StationaryHeteroskedasticNormalTanhDistribution(
            num_actions,
            hidden_layer_size[:-1],
            feature_dimension=hidden_layer_size[-1],
            prior_variance=0.75,
            stationary_activation=triangle_activation,
            layer_norm_mlp=layer_norm_policy,
        )
      elif policy_arch == PolicyArchitectures.HETSTATPRELU:
        network = StationaryHeteroskedasticNormalTanhDistribution(
            num_actions,
            hidden_layer_size[:-1],
            feature_dimension=hidden_layer_size[-1],
            prior_variance=0.75,
            stationary_activation=periodic_relu_activation,
            layer_norm_mlp=layer_norm_policy,
        )
      elif policy_arch == PolicyArchitectures.MIXHETSTATSIN:
        network = MixtureStationaryHeteroskedasticNormalTanhDistribution(
            num_actions,
            n_mixture=5,
            layers=hidden_layer_size[:-1],
            feature_dimension=hidden_layer_size[-1],
            prior_variance=0.75,
            stationary_activation=sin_cos_activation,
            layer_norm_mlp=layer_norm_policy,
        )
      elif policy_arch == PolicyArchitectures.MIXHETSTATTRI:
        network = MixtureStationaryHeteroskedasticNormalTanhDistribution(
            num_actions,
            n_mixture=5,
            layers=hidden_layer_size[:-1],
            feature_dimension=hidden_layer_size[-1],
            prior_variance=0.75,
            stationary_activation=triangle_activation,
            layer_norm_mlp=layer_norm_policy,
        )
      elif policy_arch == PolicyArchitectures.MIXHETSTATPRELU:
        network = MixtureStationaryHeteroskedasticNormalTanhDistribution(
            num_actions,
            n_mixture=5,
            layers=hidden_layer_size[:-1],
            feature_dimension=hidden_layer_size[-1],
            prior_variance=0.75,
            stationary_activation=periodic_relu_activation,
            layer_norm_mlp=layer_norm_policy,
        )
      else:
        raise ValueError('Unknown policy architecture.')

      obs = observation_encoder(obs)
      if not train_encoder:
        obs = jax.lax.stop_gradient(obs)
      return network(obs, *args, **kwargs)

    return _actor_fn

  actor_fn = _make_actor_fn(policy_architecture, policy_hidden_layer_sizes)

  prior_policy_llh = prior_policy_log_likelihood(
      spec,
      policy_architecture=PolicyArchitectures.MLP,
  )

  def _critic_fn(obs, action, train_encoder=False):
    if critic_architecture == CriticArchitectures.DOUBLE_MLP:  # Needed for SAC.
      network1 = hk.Sequential([
          hk.nets.MLP(
              list(critic_hidden_layer_sizes) + [1],
              w_init=hk.initializers.Orthogonal(),
              activation=jax.nn.elu),
      ])
      network2 = hk.Sequential([
          hk.nets.MLP(
              list(critic_hidden_layer_sizes) + [1],
              w_init=hk.initializers.Orthogonal(),
              activation=jax.nn.elu),
      ])
      obs = observation_encoder(obs)
      input_ = jnp.concatenate([obs, action], axis=-1)
      value1 = network1(input_)
      value2 = network2(input_)
      return jnp.concatenate([value1, value2], axis=-1)
    elif critic_architecture == CriticArchitectures.MLP:
      network = hk.nets.MLP(
          list(critic_hidden_layer_sizes) + [1],
          w_init=hk.initializers.Orthogonal(),
          activation=jax.nn.elu,
      )
    elif critic_architecture == CriticArchitectures.LNMLP:
      network = networks_lib.LayerNormMLP(
          list(critic_hidden_layer_sizes) + [1],
          w_init=hk.initializers.Orthogonal(),
          activation=jax.nn.elu,
      )
    elif critic_architecture == CriticArchitectures.DOUBLE_LNMLP:
      network1 = hk.Sequential([
          networks_lib.LayerNormMLP(
              list(critic_hidden_layer_sizes) + [1],
              w_init=hk.initializers.Orthogonal(),
              activation=jax.nn.elu),
      ])
      network2 = hk.Sequential([
          networks_lib.LayerNormMLP(
              list(critic_hidden_layer_sizes) + [1],
              w_init=hk.initializers.Orthogonal(),
              activation=jax.nn.elu),
      ])

      obs = observation_encoder(obs)
      if not train_encoder:
        obs = jax.lax.stop_gradient(obs)
      input_ = jnp.concatenate([obs, action], axis=-1)
      value1 = network1(input_)
      value2 = network2(input_)
      return jnp.concatenate([value1, value2], axis=-1)
    elif critic_architecture == CriticArchitectures.STATSIN:
      network = StationaryMLP(
          1,
          critic_hidden_layer_sizes[:-1],
          feature_dimension=critic_hidden_layer_sizes[-1],
          stationary_activation=sin_cos_activation,
          layer_norm_mlp=False,
      )
    elif critic_architecture == CriticArchitectures.STATTRI:
      network = StationaryMLP(
          1,
          critic_hidden_layer_sizes[:-1],
          feature_dimension=critic_hidden_layer_sizes[-1],
          stationary_activation=triangle_activation,
          layer_norm_mlp=False,
      )
    elif critic_architecture == CriticArchitectures.STATPRELU:
      network = StationaryMLP(
          1,
          critic_hidden_layer_sizes[:-1],
          feature_dimension=critic_hidden_layer_sizes[-1],
          stationary_activation=periodic_relu_activation,
          layer_norm_mlp=False,
      )
    else:
      raise ValueError('Unknown critic architecture.')

    obs = observation_encoder(obs)
    input_ = jnp.concatenate([obs, action], axis=-1)
    return network(input_)

  reward_policy_coherence = (reward_architecture == RewardArchitectures.PCSIL or
                             reward_architecture == RewardArchitectures.NCSIL)
  if reward_policy_coherence:
    assert reward_policy_coherence_alpha is not None
    assert bc_policy_architecture is not None
    assert bc_policy_hidden_layer_sizes is not None
    bc_actor_fn = _make_actor_fn(
        bc_policy_architecture, bc_policy_hidden_layer_sizes
    )

    def _reward_fn(obs, action, *args, **kwargs):
      obs = observation_encoder(obs)
      alpha = reward_policy_coherence_alpha
      log_ratio = (bc_actor_fn(obs, *args, **kwargs).log_prob(action)  # pytype: disable=attribute-error
                   - prior_policy_llh(action))
      if reward_architecture == RewardArchitectures.PCSIL:
        return alpha * log_ratio
      else:  # reward_architecture == RewardArchitectures.NCSIL
        return alpha * (log_ratio - num_actions * sil_config.MAX_REWARD)

  else:
    bc_actor_fn = actor_fn
    bc_policy_architecture = policy_architecture

    def _reward_fn(obs, action, train_encoder=False):
      if reward_architecture == RewardArchitectures.MLP:
        network = hk.nets.MLP(
            list(reward_hidden_layer_sizes) + [1],
            w_init=hk.initializers.Orthogonal(),
        )
      elif reward_architecture == RewardArchitectures.LNMLP:
        network = networks_lib.LayerNormMLP(
            list(reward_hidden_layer_sizes) + [1],
            w_init=hk.initializers.Orthogonal(),
        )
      elif reward_architecture == RewardArchitectures.PCONST:
        network = jax.vmap(lambda sa: 1.0)
      elif reward_architecture == RewardArchitectures.NCONST:
        network = jax.vmap(lambda sa: -1.0)
      else:
        raise ValueError('Unknown reward architecture.')

      obs = observation_encoder(obs)
      if not train_encoder:
        obs = jax.lax.stop_gradient(obs)
      input_ = jnp.concatenate([obs, action], axis=-1)
      return network(input_)

  policy = hk.without_apply_rng(hk.transform(actor_fn))
  bc_policy = hk.without_apply_rng(hk.transform(bc_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  reward = hk.without_apply_rng(hk.transform(_reward_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return SILNetworks(
      policy_architecture=policy_architecture,
      bc_policy_architecture=bc_policy_architecture,
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply
      ),
      critic_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
      ),
      reward_network=networks_lib.FeedForwardNetwork(
          lambda key: reward.init(key, dummy_obs, dummy_action), reward.apply
      ),
      log_prob=lambda params, actions: params.log_prob(actions),
      log_prob_prior=prior_policy_llh,
      sample=lambda params, key: params.sample(seed=key),
      # Policy eval is distribution's 'mode' (i.e. deterministic).
      sample_eval=lambda params, key: params.mode(),
      environment_specs=spec,
      reward_policy_coherence=reward_policy_coherence,
      bc_policy_network=networks_lib.FeedForwardNetwork(
          lambda key: bc_policy.init(key, dummy_obs), bc_policy.apply
      ),
  )
