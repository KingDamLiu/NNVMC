# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import spin_penalty
from ferminet import mcmc
from ferminet import networks
from ferminet import observables
from ferminet import pretrain
from ferminet import psiformer
from ferminet.networks import Network
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import utils
from ferminet.utils import writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol
import pandas as pd
import datetime
from optax._src import base
import yaml


def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))

def clip_local_energys(x):
  # 按20倍4分位距进行截断
  a = jnp.nanpercentile(x, jnp.array([25, 50, 75]), axis=0)
  upper = a[2] + 10 * (a[2] - a[0])
  lower = a[0] - 10 * (a[2] - a[0])
  return jnp.clip(x, lower, upper)

def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.
    core_electrons: mapping of element symbol to number of core electrons
      included in the pseudopotential.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """
  total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
                        for atom in molecule)
  if total_electrons != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
         atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
        for atom in molecule
    ]
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons[0], electrons[1], batch_size
  )

  return electron_positions, electron_spins


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None

def norm(obj):
  """Computes the Euclidean norm of the provided PyTree object."""
  elements_squared_norm = jax.tree_util.tree_map(
      lambda x: jnp.sum(jnp.square(x)), obj)

  return jnp.sqrt(sum(jax.tree_util.tree_leaves(elements_squared_norm)))

def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    stats = dict(
        # step=opt_state.step_counter,
        # batch_size=jnp.asarray(total_batch_size, dtype=jnp.int32),
        # data_seen=opt_state.data_seen,
        loss=loss,
        new_loss=loss,
    )

    # if _value_func_has_aux:
    stats["aux"] = aux_data

    # if _include_norms_in_stats:
    stats["param_norm"] = norm(params)
    stats["grad_norm"] = norm(grad)
    stats["precon_grad_norm"] = norm(grad)
    stats["update_norm"] = norm(updates)
    stats['learning_rate'] = (None, None)
    stats['momentum'] = (None, None)

    return params, opt_state, stats

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove

  return step


def make_kfac_training_step(
    mcmc_step,
    damping: float,
    optimizer: kfac_jax.Optimizer,
    reset_if_nan: bool = False) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))
  # Due to some KFAC cleverness related to donated buffers, need to do this
  # to make state resettable
  copy_tree = constants.pmap(
      functools.partial(jax.tree_util.tree_map,
                        lambda x: (1.0 * x).astype(x.dtype)))
    
  # 初始化缓存区，保存参数和状态
  old_params_buffer = []
  old_state_buffer = []
  old_stats_buffer = []
  t = 0  # 计数器
  
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.Optimizer.State,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.
    nonlocal t
    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    # Optimization step
    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
    )

    # 每 10 次迭代保存一次参数、状态和统计信息
    if reset_if_nan and t % 10 == 0:
        # 深拷贝当前的参数、状态和统计信息，避免后续修改影响缓存
        old_params_buffer.append(copy_tree(new_params))
        old_state_buffer.append(copy_tree(new_state))
        # 由于 stats 是字典，需要深拷贝
        old_stats_buffer.append(copy_tree(stats))
        # 确保缓存区只保存两次信息，超过则移除最早的
        if len(old_params_buffer) > 2:
            old_params_buffer.pop(0)
            old_state_buffer.pop(0)
            old_stats_buffer.pop(0)

    # 检查损失是否为 NaN
    if reset_if_nan and jnp.isnan(stats['loss']).any():
        # 使用第一次保存的参数、状态和统计信息进行恢复
        if old_params_buffer:
            new_params = old_params_buffer[0]
            new_state = old_state_buffer[0]
            stats = old_stats_buffer[0]
            print(f"第 {t} 次迭代损失为 NaN,恢复到第 {t - (len(old_params_buffer) * 10)} 次迭代的参数、状态和统计信息。")
        else:
            # 如果缓存区为空，可能是在前 10 次迭代内发生的异常
            print(f"第 {t} 次迭代损失为 NaN,但无可用的缓存进行恢复。")
            # 可以根据需要决定是抛出异常还是继续训练
    t = t+1
    return data, new_params, new_state, stats, pmove

  return step

def init_config(cfg):
    # 设备日志记录和设置
    num_devices = jax.local_device_count()
    num_hosts = jax.device_count() // num_devices
    num_states = cfg.system.get('states', 0) or 1  # 避免0/1的混淆
    logging.info('Starting QMC with %i XLA devices per host '
                 'across %i hosts.', num_devices, num_hosts)
    if cfg.batch_size % (num_devices * num_hosts) != 0:
        raise ValueError('Batch size must be divisible by number of devices, '
                         f'got batch size {cfg.batch_size} for '
                         f'{num_devices * num_hosts} devices.')
    host_batch_size = cfg.batch_size // num_hosts  # 每个主机的批量大小
    total_host_batch_size = host_batch_size * num_states
    device_batch_size = host_batch_size // num_devices  # 每个设备的批量大小
    data_shape = (num_devices, device_batch_size)

    # 检查并转换pyscf分子为内部表示形式
    if cfg.system.pyscf_mol:
        cfg.update(
            system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

    # 将分子配置转换为原子位置和电荷的数组
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    nspins = cfg.system.electrons

    # 为每个行走者生成原子配置
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

    # 设置随机种子
    if cfg.debug.deterministic:
        seed = 23
    else:
        seed = jnp.asarray([1e6 * time.time()])
        seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    key = jax.random.PRNGKey(seed)

    # 提取因赝势移除的每个原子周围的自旋电子数
    if cfg.system.pyscf_mol:
        cfg.system.pyscf_mol.build()
        core_electrons = {
            atom: ecp_table[0]
            for atom, ecp_table in cfg.system.pyscf_mol._ecp.items()  # pylint: disable=protected-access
        }
        ecp = cfg.system.pyscf_mol.ecp
    else:
        ecp = {}
        core_electrons = {}

    return (cfg, key, num_devices, num_hosts, num_states, host_batch_size,
            total_host_batch_size, device_batch_size, data_shape, atoms,
            charges, nspins, batch_atoms, batch_charges, core_electrons, ecp)

def init_reference_wavefunction(cfg, key, charges, nspins, core_electrons, ecp):
    if cfg.pretrain.iterations > 0:
        hartree_fock = pretrain.get_hf(
            pyscf_mol=cfg.system.get('pyscf_mol'),
            method = cfg.pretrain.method,
            molecule=cfg.system.molecule,
            nspins=nspins,
            restricted=False,
            basis=cfg.pretrain.basis,
            ecp=ecp,
            core_electrons=core_electrons,
            states=cfg.system.states,
            excitation_type=cfg.pretrain.get('excitation_type', 'ordered'))
        # broadcast the result of PySCF from host 0 to all other hosts
        hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(
            hartree_fock.mean_field.mo_coeff
        )
    return hartree_fock


def init_wavefunction_single(cfg, key, charges, nspins, core_electrons, ecp):
    # 初始化特征层
    if cfg.network.make_feature_layer_fn:
        feature_layer_module, feature_layer_fn = (
            cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
        feature_layer_module = importlib.import_module(feature_layer_module)
        make_feature_layer: networks.MakeFeatureLayer = getattr(
            feature_layer_module, feature_layer_fn
        )
        feature_layer = make_feature_layer(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs)
    else:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
        )

    # 初始化包络函数
    if cfg.network.make_envelope_fn:
        envelope_module, envelope_fn = (
            cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
        envelope_module = importlib.import_module(envelope_module)
        make_envelope = getattr(envelope_module, envelope_fn)
        envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
    else:
        envelope = envelopes.make_isotropic_envelope()

    # 初始化网络
    use_complex = cfg.network.get('complex', False)
    if cfg.network.network_type == 'ferminet':
        network = networks.make_fermi_net(
            nspins,
            charges,
            ndim=cfg.system.ndim,
            determinants=cfg.network.determinants,
            states=cfg.system.states,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.get('jastrow', 'default'),
            bias_orbitals=cfg.network.bias_orbitals,
            full_det=cfg.network.full_det,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            complex_output=use_complex,
            **cfg.network.ferminet,
        )
    elif cfg.network.network_type == 'psiformer':
        network = psiformer.make_fermi_net(
            nspins,
            charges,
            ndim=cfg.system.ndim,
            determinants=cfg.network.determinants,
            states=cfg.system.states,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.get('jastrow', 'default'),
            bias_orbitals=cfg.network.bias_orbitals,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            complex_output=use_complex,
            **cfg.network.psiformer,
        )
    else:
        raise ValueError(f"Unknown network type: {cfg.network.network_type}")

    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply

    # 通常只需要log|psi(x)|
    if cfg.system.get('states', 0):
        network_mcmc = networks.make_state_diag(signed_network,
                                                       cfg.system.states)
        if cfg.optim.objective == 'vmc_overlap':
            logabs_network = networks.make_state_trace(signed_network,
                                                       cfg.system.states)
        else:
            logabs_network = utils.select_output(
                networks.make_total_ansatz(signed_network,
                                           cfg.system.get('states', 0),
                                           complex_output=use_complex), 1)
    else:
        logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )  # 批量网络
    if cfg.mcmc.target == 'Psi':
        batch_network_mcmc = batch_network
    else:
        batch_network_mcmc = jax.vmap(
            network_mcmc, in_axes=(None, 0, 0, 0, 0), out_axes=0
        )  # batched network for MCMC
    # 当计算复杂波函数的能量梯度时，需要log(psi)而不是log(|psi|)
    if cfg.system.get('states', 0):
        log_network_for_loss = None
        if cfg.optim.objective == 'vmc_overlap':
            # 在惩罚方法的情况下，我们实际上需要所有输出来计算梯度
            log_network_for_loss = networks.make_state_matrix(signed_network,
                                                              cfg.system.states)
            def log_network(*args, **kwargs):
                phase, mag = log_network_for_loss(*args, **kwargs)
                return mag + 1.j * phase
        else:
            def log_network(*args, **kwargs):
                if not use_complex:
                    raise ValueError('This function should never be used if the '
                                     'wavefunction is real-valued.')
                meta_net = networks.make_total_ansatz(signed_network,
                                                      cfg.system.get('states', 0),
                                                      complex_output=True)
                phase, mag = meta_net(*args, **kwargs)
                return mag + 1.j * phase
    else:
        def log_network(*args, **kwargs):
            if not use_complex:
                raise ValueError('This function should never be used if the '
                                 'wavefunction is real-valued.')
            phase, mag = signed_network(*args, **kwargs)
            return mag + 1.j * phase
    networks_list_ = []
    return (params, network, networks_list_, signed_network, logabs_network,
            batch_network, log_network, batch_network_mcmc, log_network_for_loss, key)

def init_wavefunction(cfg, key, charges, nspins, core_electrons, ecp):
    if cfg.network.type=='multi':
        return init_wavefunction_multi(cfg, key, charges, nspins, core_electrons, ecp)
    else:
        return init_wavefunction_single(cfg, key, charges, nspins, core_electrons, ecp)

def init_wavefunction_multi(cfg, key, charges, nspins, core_electrons, ecp):
    # 初始化特征层
    if cfg.network.make_feature_layer_fn:
        feature_layer_module, feature_layer_fn = (
            cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
        feature_layer_module = importlib.import_module(feature_layer_module)
        make_feature_layer: networks.MakeFeatureLayer = getattr(
            feature_layer_module, feature_layer_fn
        )
        feature_layer = make_feature_layer(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs)
    else:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=cfg.system.electrons,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
        )

    # 初始化包络函数
    if cfg.network.make_envelope_fn:
        envelope_module, envelope_fn = (
            cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
        envelope_module = importlib.import_module(envelope_module)
        make_envelope = getattr(envelope_module, envelope_fn)
        envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
    else:
        envelope = envelopes.make_isotropic_envelope()

    # 初始化网络
    use_complex = cfg.network.get('complex', False)
    num_states = cfg.system.get('states', 0) or 1  # 避免 0/1 混淆
    num_states_update = cfg.system.states_update
    networks_list = []
    params_list = []
    for state in range(num_states):
        if cfg.network.network_type == 'ferminet':
            network = networks.make_fermi_net(
                nspins,
                charges,
                ndim=cfg.system.ndim,
                determinants=cfg.network.determinants,
                states=1,  # 每个网络只对应一个状态
                envelope=envelope,
                feature_layer=feature_layer,
                jastrow=cfg.network.get('jastrow', 'default'),
                bias_orbitals=cfg.network.bias_orbitals,
                full_det=cfg.network.full_det,
                rescale_inputs=cfg.network.get('rescale_inputs', False),
                complex_output=use_complex,
                **cfg.network.ferminet,
            )
        elif cfg.network.network_type == 'psiformer':
            network = psiformer.make_fermi_net(
                nspins,
                charges,
                ndim=cfg.system.ndim,
                determinants=cfg.network.determinants,
                states=1,  # 每个网络只对应一个状态
                envelope=envelope,
                feature_layer=feature_layer,
                jastrow=cfg.network.get('jastrow', 'default'),
                bias_orbitals=cfg.network.bias_orbitals,
                rescale_inputs=cfg.network.get('rescale_inputs', False),
                complex_output=use_complex,
                **cfg.network.psiformer,
            )
        else:
            raise ValueError(f"Unknown network type: {cfg.network.network_type}")

        key, subkey = jax.random.split(key)
        params = network.init(subkey)
        networks_list.append(network)
        params_list.append(params)
    
    # 将参数打包成一个字典，键为 'state_0', 'state_1', ...
    params = {'state_{}'.format(i): p for i, p in enumerate(params_list)}
    params = kfac_jax.utils.replicate_all_local_devices(params)
    
    cfg.optim.frozen_states_indices = [] # ['state_0', 'state_1', 'state_2', 'state_3']
    # 定义 signed_network 函数，调用正确的网络
    def signed_network(params, pos, spins, atoms, charges, **kwargs): 
        # params 是一个字典，键为 'state_0', 'state_1', ...
        # pos 形状为 (n_electrons * ndim)
        signs = []
        logabs = []
        for i in range(num_states):
            p = params['state_{}'.format(i)]
            net = networks_list[i]
            sign, logab = net.apply(p, pos, spins, atoms, charges, **kwargs)
            signs.append(sign)
            logabs.append(logab)
        # 将输出堆叠
        signs = jnp.concatenate(signs, axis=0)
        logabs = jnp.concatenate(logabs, axis=0)
        return signs, logabs

    networks_list_ = [net.apply for net in networks_list]

    def signed_network_update(params, pos, spins, atoms, charges, **kwargs): 
        # params 是一个字典，键为 'state_0', 'state_1', ...
        # pos 形状为 (n_electrons * ndim)
        signs = []
        logabs = []
        for i in range(num_states_update):
            p = params['state_{}'.format(i)]
            net = networks_list[i]
            sign, logab = net.apply(p, pos, spins, atoms, charges, **kwargs)
            signs.append(sign)
            logabs.append(logab)
        # 将输出堆叠
        signs = jnp.concatenate(signs, axis=0)
        logabs = jnp.concatenate(logabs, axis=0)
        return signs, logabs
    
    def signed_network_old(params, pos, spins, atoms, charges, **kwargs): 
        # params 是一个字典，键为 'state_0', 'state_1', ...
        # pos 形状为 (batch_size, n_electrons * ndim)
        signs = []
        logabs = []
        pos_ = jnp.reshape(pos, [num_states, -1])
        spins_ = jnp.reshape(spins, [num_states, -1])
        for i in range(num_states_update, num_states):
            for j in range(num_states_update):
                p = params['state_{}'.format(i)]
                net = networks_list[i]
                sign, logab = net.apply(p, pos_[j], spins_[j], atoms, charges, **kwargs)
                signs.append(sign)
                logabs.append(logab)
        # 将输出堆叠
        signs = jnp.concatenate(signs, axis=0)
        logabs = jnp.concatenate(logabs, axis=0)
        return signs, logabs
    
    def network_diag_update(params, pos, spins, atoms, charges, **kwargs):
        # params 是一个字典，键为 'state_0', 'state_1', ...
        # pos 形状为 (batch_size, n_electrons * ndim)
        pos_ = jnp.reshape(pos, [num_states, -1])
        spins_ = jnp.reshape(spins, [num_states, -1])
        signs = []
        logabs = []
        print('network_diag ', pos.shape, spins.shape, atoms.shape, charges.shape)
        for i in range(num_states_update):
            p = params['state_{}'.format(i)]
            net = networks_list[i]
            sign, logab = net.apply(p, pos_[i], spins_[i], atoms, charges, **kwargs)
            signs.append(sign)
            logabs.append(logab)
        # 将输出堆叠
        signs = jnp.concatenate(signs, axis=0)
        logabs = jnp.concatenate(logabs, axis=0)
        return signs, logabs

    def network_diag_old(params, pos, spins, atoms, charges, **kwargs):
        # params 是一个字典，键为 'state_0', 'state_1', ...
        # pos 形状为 (batch_size, n_electrons * ndim)
        pos_ = jnp.reshape(pos, [num_states, -1])
        spins_ = jnp.reshape(spins, [num_states, -1])
        signs = []
        logabs = []
        print('network_diag ', pos.shape, spins.shape, atoms.shape, charges.shape)
        for i in range(num_states_update, num_states):
            p = params['state_{}'.format(i)]
            net = networks_list[i]
            sign, logab = net.apply(p, pos_[i], spins_[i], atoms, charges, **kwargs)
            signs.append(sign)
            logabs.append(logab)
        # 将输出堆叠
        signs = jnp.concatenate(signs, axis=0)
        logabs = jnp.concatenate(logabs, axis=0)
        return signs, logabs

    def network_mcmc(params, pos, spins, atoms, charges, **kwargs): 
        # params 是一个字典，键为 'state_0', 'state_1', ...
        # pos 形状为 (batch_size, n_electrons * ndim)
        pos_ = jnp.reshape(pos, [num_states, -1])
        spins_ = jnp.reshape(spins, [num_states, -1])
        signs = []
        logabs = []
        for i in range(num_states):
            p = params['state_{}'.format(i)]
            net = networks_list[i]
            sign, logab = net.apply(p, pos_[i], spins_[i], atoms, charges, **kwargs)
            signs.append(sign)
            logabs.append(logab)
        # 将输出堆叠
        signs = jnp.concatenate(signs, axis=0)
        logabs = jnp.concatenate(logabs, axis=0)
        return logabs

    # 定义 signed_network 函数，调用正确的网络
    def network_orbitals_apply(params, pos, spins, atoms, charges, **kwargs):
        orbitals = []
        single_obritals = []
        cfg.pretrain.psi_flag = 'matrix'
        if cfg.pretrain.psi_flag == 'diag':
            pos_ = jnp.reshape(pos, [num_states, -1])
            spins_ = jnp.reshape(spins, [num_states, -1])
            for i in range(num_states):
                p = params['state_{}'.format(i)]
                net = networks_list[i]
                out,sing = net.orbitals(p, pos_[i], spins_[i], atoms, charges, **kwargs)
                orbitals.append(out)
                single_obritals.append(sing)
        elif cfg.pretrain.psi_flag == 'matrix':
            for i in range(num_states):
                p = params['state_{}'.format(i)]
                net = networks_list[i]
                out = net.orbitals(p, pos, spins, atoms, charges, **kwargs)
                orbitals.append(out)

        # 将输出堆叠
        orbitals = [jnp.stack(outputs, axis=0) for outputs in zip(*orbitals)]  # 形状为 (num_states, batch_size, ...)
        orbitals = [jnp.reshape(outputs, (outputs.shape[0]*outputs.shape[1],)+outputs.shape[2:])  for outputs in orbitals]

        return orbitals
    
    network = Network(
      options=network.options, init=None, apply=None, orbitals=network_orbitals_apply
    )

    # 通常只需要log|psi(x)|
    if cfg.system.get('states', 0):
        if cfg.optim.objective == 'vmc_overlap':
            logabs_network = networks.make_state_trace(signed_network,
                                                       cfg.system.states)
        else:
            logabs_network = utils.select_output(
                networks.make_total_ansatz(signed_network,
                                           cfg.system.get('states', 0),
                                           complex_output=use_complex), 1)
    else:
        logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    batch_network_mcmc = jax.vmap(
        network_mcmc, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )  # 批量网络
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )  # 批量网络

    log_network_for_loss = networks.make_state_matrix(signed_network,
                                                        cfg.system.states)

    # 当计算复杂波函数的能量梯度时，需要log(psi)而不是log(|psi|)
    if cfg.system.get('states', 0):
        if cfg.optim.objective == 'vmc_overlap':
            # 在惩罚方法的情况下，我们实际上需要所有输出来计算梯度
            log_network_for_loss = networks.make_state_matrix(signed_network,
                                                              cfg.system.states)
            def log_network(*args, **kwargs):
                phase, mag = log_network_for_loss(*args, **kwargs)
                return mag + 1.j * phase
        else:
            def log_network(*args, **kwargs):
                if not use_complex:
                    raise ValueError('This function should never be used if the '
                                     'wavefunction is real-valued.')
                meta_net = networks.make_total_ansatz(signed_network,
                                                      cfg.system.get('states', 0),
                                                      complex_output=True)
                phase, mag = meta_net(*args, **kwargs)
                return mag + 1.j * phase
    else:
        def log_network(*args, **kwargs):
            if not use_complex:
                raise ValueError('This function should never be used if the '
                                 'wavefunction is real-valued.')
            phase, mag = signed_network(*args, **kwargs)
            return mag + 1.j * phase

    return (params, network, networks_list_, signed_network, logabs_network,
            batch_network, log_network, batch_network_mcmc, log_network_for_loss, key)

def init_data(cfg, key, data_shape, total_host_batch_size, batch_atoms,
              batch_charges, core_electrons):

    key, subkey = jax.random.split(key)
    # 确保每个主机上的数据初始化不同
    subkey = jax.random.fold_in(subkey, jax.process_index())
    # 创建电子状态（位置和自旋）
    pos, spins = init_electrons(
        subkey,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size=total_host_batch_size,
        init_width=cfg.mcmc.init_width,
        core_electrons=core_electrons,
    )
    # 对于激发态，每个设备有一批行走者，每个行走者是 nstates * nelectrons
    pos = jnp.reshape(pos, data_shape + (-1,))
    pos = kfac_jax.utils.broadcast_all_local_devices(pos)
    spins = jnp.reshape(spins, data_shape + (-1,))
    spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    data = networks.FermiNetData(
        positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
    )

    return data

def init_datas(cfg, key, data_shape, total_host_batch_size, batch_atoms,
              batch_charges, core_electrons, params, host_batch_size):
    # 存在设置检查点的情况
    ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)
    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path+'/vmctrain/')
    ckpt_pretrained_path = checkpoint.create_save_path(cfg.log.save_path+'/pretrain/')

    search_paths = [
        (ckpt_save_path, 'ckpt_save_path'),
        (ckpt_pretrained_path, 'ckpt_pretrained_path'),
    ]

    ckpt_restore_filename, ckpt_restore_source = None, None
    for path, source_name in search_paths:
        found = checkpoint.find_last_checkpoint(path)
        if found is not None:
            ckpt_restore_filename = found
            ckpt_restore_source = source_name
            break

    
    # 将 ConfigDict 转换为普通的 Python 字典并保存为 YAML 文件
    config_dict = cfg.to_dict()
    with open(cfg.log.save_path + "/config.yaml", "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    # 分四种情况：
    # 有设置检查点，有前置数据，有预训练数据

    if ckpt_restore_filename:
        (t_init,
         data,
         params,
         opt_state_ckpt,
         mcmc_width_ckpt,
         density_state_ckpt) = checkpoint.restore(
             ckpt_restore_filename, host_batch_size)
        # if data is None or ckpt_restore_source == 'ckpt_pretrained_path':
        #     logging.info('Old data is not available, initializing new data.')
        #     data = init_data(cfg, key, data_shape, total_host_batch_size, batch_atoms,
        #         batch_charges, core_electrons)
        if ckpt_restore_source == 'ckpt_pretrained_path':
            t_init = 0
            opt_state_ckpt = None
            mcmc_width_ckpt = None
            density_state_ckpt = None
    else:
        logging.info('No checkpoint found. Training new model.')
        data = init_data(cfg, key, data_shape, total_host_batch_size, batch_atoms,
            batch_charges, core_electrons)

        t_init = 0
        opt_state_ckpt = None
        mcmc_width_ckpt = None
        density_state_ckpt = None

    return (t_init, data, params, opt_state_ckpt, mcmc_width_ckpt,
            density_state_ckpt, ckpt_save_path, key)

def init_log(cfg, ckpt_save_path, signed_network, nspins, data,
             density_state_ckpt, key):
    # 设置日志和观测量
    train_schema = ['step', 'time', 'energy', 'var', 'ewmean', 'ewvar', 'pmove',
                    'grad_norm', 'learning_rate', 'momentum', 'param_norm',
                    'precon_grad_norm', 'update_norm']

    if cfg.system.states:
        energy_matrix_file = open(
            os.path.join(ckpt_save_path, 'energy_matrix.npy'), 'ab')
        energy_matrix_std_file = open(
            os.path.join(ckpt_save_path, 'energy_matrix_std.npy'), 'ab')
        energy_matrix_clip_file = open(
            os.path.join(ckpt_save_path, 'energy_matrix_clip.npy'), 'ab')
        energy_matrix_clip_std_file = open(
            os.path.join(ckpt_save_path, 'energy_matrix_clip_std.npy'), 'ab')
    else:
        energy_matrix_file = energy_matrix_std_file = \
            energy_matrix_clip_file = energy_matrix_clip_std_file = None

    observable_fns = {}
    observable_states = {}  # 仅对密度矩阵相关
    if cfg.observables.s2:
        observable_fns['s2'] = observables.make_s2(
            signed_network,
            nspins,
            states=cfg.system.states)
        observable_states['s2'] = None
        train_schema += ['s2']
        if cfg.system.states:
            s2_matrix_file = open(
                os.path.join(ckpt_save_path, 's2_matrix.npy'), 'ab')
        else:
            s2_matrix_file = None
    else:
        s2_matrix_file = None

    if cfg.observables.dipole:
        observable_fns['dipole'] = observables.make_dipole(
            signed_network,
            states=cfg.system.states)
        observable_states['dipole'] = None
        train_schema += ['mu_x', 'mu_y', 'mu_z']
        if cfg.system.states:
            dipole_matrix_file = open(
                os.path.join(ckpt_save_path, 'dipole_matrix.npy'), 'ab')
        else:
            dipole_matrix_file = None
    else:
        dipole_matrix_file = None

    # 在创建密度矩阵函数之前执行此操作，因为这是一个特例
    observable_fns = observables.make_observable_fns(observable_fns)

    if cfg.observables.density:
        (observable_states['density'],
         density_update,
         observable_fns['density']) = observables.make_density_matrix(
             signed_network, data.positions, cfg, density_state_ckpt)
        # 因为即使没有激发态，密度矩阵也可能非常大，所以我们总是将其直接保存到.npy文件，而不是写入CSV
        density_matrix_file = open(
            os.path.join(ckpt_save_path, 'density_matrix.npy'), 'ab')
        # 仅针对密度矩阵函数的自定义pmap
        pmap_density_axes = observables.DensityState(t=None,
                                                     positions=0,
                                                     probabilities=0,
                                                     move_width=0,
                                                     pmove=None,
                                                     mo_coeff=None)
        pmap_fn = constants.pmap(observable_fns['density'],
                                 in_axes=(0, 0, pmap_density_axes))
        observable_fns['density'] = lambda *a, **kw: pmap_fn(*a, **kw).mean(0)
    else:
        density_update = None
        density_matrix_file = None

    # 初始化完成。我们现在希望在每个设备上有不同的PRNG流。将密钥分片到设备上
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    return (train_schema, observable_fns, observable_states, energy_matrix_file,
            energy_matrix_std_file, energy_matrix_clip_file,
            energy_matrix_clip_std_file, s2_matrix_file, dipole_matrix_file,
            density_matrix_file, sharded_key, density_update)

def init_mcmc_step(cfg, batch_network_mcmc, device_batch_size, atoms, num_states):
    if cfg.mcmc.target == 'Psi':
        blocks = cfg.mcmc.blocks * num_states
    else:
        blocks = 1
    atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
    mcmc_step = mcmc.make_mcmc_step(
        batch_network_mcmc,
        device_batch_size,
        steps=cfg.mcmc.steps,
        atoms=atoms_to_mcmc,
        blocks=blocks,
    )
    return mcmc_step

def init_local_energy_fn(cfg, signed_network, networks_list, charges, nspins, use_complex):
    laplacian_method = cfg.optim.get('laplacian', 'default')
    state_specific = 0
    if cfg.optim.objective == 'vmc_overlap':
        state_specific = 2
    elif cfg.optim.objective == 'vmc_ipf':
        state_specific = 2
    if cfg.system.make_local_energy_fn:
        if laplacian_method != 'default':
            raise NotImplementedError(f'Laplacian method {laplacian_method}'
                                      'not yet supported by custom local energy fns.')
        if cfg.optim.objective == 'vmc_overlap':
            raise NotImplementedError('Overlap penalty not yet supported for custom'
                                      'local energy fns.')
        local_energy_module, local_energy_fn_name = (
            cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
        local_energy_module = importlib.import_module(local_energy_module)
        make_local_energy = getattr(local_energy_module, local_energy_fn_name)
        local_energy_fn = make_local_energy(
            f=signed_network,
            charges=charges,
            nspins=nspins,
            use_scan=False,
            states=cfg.system.get('states', 0),
            **cfg.system.make_local_energy_kwargs)
    elif 0:
        pp_symbols = cfg.system.get('pp', {'symbols': None}).get('symbols')
        local_energy_fn = hamiltonian.local_energy(
            f=signed_network,
            charges=charges,
            nspins=nspins,
            use_scan=False,
            complex_output=use_complex,
            laplacian_method=laplacian_method,
            states=cfg.system.get('states', 0),
            state_specific = state_specific,
            pp_type=cfg.system.get('pp', {'type': 'ccecp'}).get('type'),
            pp_symbols=pp_symbols if cfg.system.get('use_pp') else None)
    else:
        pp_symbols = cfg.system.get('pp', {'symbols': None}).get('symbols')
        local_energy_fn = hamiltonian.local_energy_k(
            f=networks_list,
            charges=charges,
            nspins=nspins,
            use_scan=False,
            complex_output=use_complex,
            laplacian_method=laplacian_method,
            states=cfg.system.get('states', 0),
            states_update = cfg.system.states_update,
            state_specific = state_specific,
            pp_type=cfg.system.get('pp', {'type': 'ccecp'}).get('type'),
            pp_symbols=pp_symbols if cfg.system.get('use_pp') else None)
    return local_energy_fn

def init_evaluate_loss(cfg, log_network, networks_list, log_network_for_loss, logabs_network, signed_network, local_energy_fn, use_complex):
    # local_s2_fns = []
    # for i in range(cfg.system.states):
    #     local_s2_fn = observables.make_s2(
    #         networks_list[0],
    #         nspins=cfg.system.electrons,
    #         states=None)
    #     local_s2_fns.append(local_s2_fn)
    # local_energy = local_energy_fn
    # # if cfg.optim.get('spin_energy', 0.0) > 0.0:
    # #     # 最小化 <H + c * S^2> 而不是仅仅 <H>
    # #     # 创建一个新的局部能量函数，该函数取局部能量和局部自旋大小的加权和
    # #     def local_energy_and_s2_fn(params, keys, data):
    # #         local_energy, aux_data = local_energy_fn(params, keys, data)
    # #         p = params['state_{}'.format(cfg.system.states-1)]
    # #         data_ = networks.FermiNetData(
    # #             positions=data.positions[0],
    # #             spins=data.spins[0],
    # #             atoms=data.atoms,
    # #             charges=data.charges,
    # #         )
    # #         s2 = local_s2_fn(p, data_, None)
    # #         weight = cfg.optim.get('spin_energy', 0.0)
    # #         if 0:
    # #             aux_data = aux_data + weight * s2
    # #             local_energy_and_s2 = local_energy + weight * jnp.trace(s2)
    # #         else:
    # #             local_energy_and_s2 = local_energy + weight * s2
    # #         return local_energy_and_s2, aux_data
    # #     local_energy = local_energy_and_s2_fn
    # # else:
    # #     local_energy = local_energy_fn
    local_energy = local_energy_fn
    spin_loss = spin_penalty.make_spin_penalty(
        networks_list[-1],
        nspins=cfg.system.electrons,
        with_spin_grad=True)

    if cfg.optim.objective == 'vmc':
        evaluate_loss = qmc_loss_functions.make_loss(
            log_network if use_complex else logabs_network,
            local_energy,
            clip_local_energy=cfg.optim.clip_local_energy,
            clip_from_median=cfg.optim.clip_median,
            center_at_clipped_energy=cfg.optim.center_at_clip,
            complex_output=use_complex,
            max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
        )
    elif cfg.optim.objective == 'wqmc':
        evaluate_loss = qmc_loss_functions.make_wqmc_loss(
            log_network if use_complex else logabs_network,
            local_energy,
            clip_local_energy=cfg.optim.clip_local_energy,
            clip_from_median=cfg.optim.clip_median,
            center_at_clipped_energy=cfg.optim.center_at_clip,
            complex_output=use_complex,
            max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
            vmc_weight=cfg.optim.get('vmc_weight', 1.0)
        )
    elif cfg.optim.objective == 'vmc_overlap':
        if not cfg.system.states:
            raise ValueError('Overlap penalty only works with excited states')
        if cfg.optim.overlap.weights is None:
            overlap_weight = tuple([1./(1.+x) for x in range(cfg.system.states)])
            overlap_weight = tuple([x/sum(overlap_weight) for x in overlap_weight])
        else:
            assert len(cfg.optim.overlap.weights) == cfg.system.states
            overlap_weight = cfg.optim.overlap.weights
        evaluate_loss = qmc_loss_functions.make_energy_overlap_loss(
            log_network_for_loss,
            local_energy,
            clip_local_energy=cfg.optim.clip_local_energy,
            clip_from_median=cfg.optim.clip_median,
            center_at_clipped_energy=cfg.optim.center_at_clip,
            overlap_penalty=cfg.optim.overlap.penalty,
            overlap_weight=overlap_weight,
            complex_output=use_complex,
            max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0))
    elif cfg.optim.objective == 'vmc_ipf':
        if not cfg.system.states:
            raise ValueError('Overlap penalty only works with excited states')
        if cfg.optim.overlap.weights is None:
            overlap_weight = tuple([1./(1.+x) for x in range(cfg.system.states)])
            overlap_weight = tuple([x/sum(overlap_weight) for x in overlap_weight])
        else:
            assert len(cfg.optim.overlap.weights) == cfg.system.states
            overlap_weight = cfg.optim.overlap.weights
        evaluate_loss = qmc_loss_functions.make_energy_overlap_loss_iterative(
            log_network_for_loss,
            networks_list,
            local_energy,
            states=cfg.system.states,
            states_update = cfg.system.states_update,
            clip_local_energy=cfg.optim.clip_local_energy,
            clip_from_median=cfg.optim.clip_median,
            center_at_clipped_energy=cfg.optim.center_at_clip,
            overlap_penalty=cfg.optim.overlap.penalty,
            overlap_weight=overlap_weight,
            complex_output=use_complex,
            max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0))
    else:
        raise ValueError(f'Not a recognized objective: {cfg.optim.objective}')
    def total_loss(
        params: networks.ParamTree,
        key: chex.PRNGKey,
        data: networks.FermiNetData,
    ):
        """Evaluates the total loss for a batch of configurations."""
        energy, aux_data = evaluate_loss(params, key, data)
        p = params[list(params.keys())[-1]]
        data_ = networks.FermiNetData(
            positions=data.positions[:, -1],
            spins=data.spins[:, -1],
            atoms=data.atoms,
            charges=data.charges,
        )
        spin, aux_data_spin = spin_loss(p, key, data_)
        aux_data.spin_data = aux_data_spin
        return energy + spin, aux_data
    
    return evaluate_loss

def init_learning_rate_schedule(cfg):
    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)
    return learning_rate_schedule

def init_optimizer(cfg, evaluate_loss, learning_rate_schedule, params,
                   sharded_key, data, opt_state_ckpt):
    if cfg.optim.optimizer == 'none':
        optimizer = None
        opt_state = None
    elif cfg.optim.optimizer == 'adam':
        optimizer = optax.chain(
            optax.scale_by_adam(**cfg.optim.adam),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.))
        opt_state = jax.pmap(optimizer.init)(params)
        opt_state = opt_state_ckpt or opt_state
    elif cfg.optim.optimizer == 'lamb':
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(eps=1e-7),
            optax.scale_by_trust_ratio(),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1))
        opt_state = jax.pmap(optimizer.init)(params)
        opt_state = opt_state_ckpt or opt_state
    elif cfg.optim.optimizer == 'kfac':
        val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
        optimizer = kfac_jax.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            value_func_has_rng=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode='fisher_exact',
            include_norms_in_stats = True,
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME,
            auto_register_kwargs=dict(
                graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
            ),
            # debug=True
        )
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, data)
        opt_state = opt_state_ckpt or opt_state
    else:
        raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')
    return optimizer, opt_state, sharded_key

def run_training_loop(
    cfg, t_init, data, params, opt_state, mcmc_step, evaluate_loss, optimizer,
    sharded_key, ckpt_save_path, writer_manager, observable_fns, observable_states,
    mcmc_width_ckpt, density_update, train_schema, energy_matrix_file,
    energy_matrix_std_file, energy_matrix_clip_file, energy_matrix_clip_std_file,
    s2_matrix_file, dipole_matrix_file, density_matrix_file, data_shape
):
    # 初始化变量
    time_of_last_ckpt = time.time()
    weighted_stats = None
    num_resets = 0  # 用于 reset_if_nan 为真时
    if mcmc_width_ckpt is not None:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
    else:
        mcmc_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.move_width))
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    if cfg.mcmc.target == 'diag':
        data.positions = jnp.reshape(data.positions, data_shape + (cfg.system.states, -1,))
        data.spins = jnp.reshape(data.spins, data_shape + (cfg.system.states, -1,))
    # 执行burn-in步骤
    if t_init == 0:
        logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
        burn_in_step = make_training_step(
            mcmc_step=mcmc_step, optimizer_step=null_update)
        for _ in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, *_ = burn_in_step(
                data,
                params,
                state=None,
                key=subkeys,
                mcmc_width=mcmc_width)
        logging.info('Completed burn-in MCMC steps')
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, _ = ptotal_energy(params, subkeys, data)
        logging.info('Initial energy: %03.4f E_h', initial_energy[0])

    # 设置训练步骤函数
    if not optimizer:
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_loss_step(evaluate_loss))
    elif isinstance(optimizer, optax.GradientTransformation):
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
            reset_if_nan=cfg.optim.reset_if_nan)
    elif isinstance(optimizer, kfac_jax.Optimizer):
        step = make_kfac_training_step(
            mcmc_step=mcmc_step,
            damping=cfg.optim.kfac.damping,
            optimizer=optimizer,
            reset_if_nan=cfg.optim.reset_if_nan)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')

    # 主训练循环
    out_file_name = os.path.join(ckpt_save_path, 'train_stats'+ datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.csv')
    start_time = time.time()
    if writer_manager is None:
        writer_manager = writers.Writer(
            name='train_stats',
            schema=train_schema,
            directory=ckpt_save_path,
            iteration_key=None,
            log=False)
    with writer_manager as writer:
        for t in range(t_init, cfg.optim.iterations):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, opt_state, stats, pmove = step(
                data,
                params,
                opt_state,
                subkeys,
                mcmc_width)

            # 处理统计信息和日志记录
            loss = stats['loss'][0]
            aux_data = stats['aux']
            weighted_stats = statistics.exponentialy_weighted_stats(
                alpha=0.1, observation=loss, previous_stats=weighted_stats)
            pmove = pmove[0]

            # 更新观测量
            observable_data = {
                key: fn(params, data, observable_states[key])
                for key, fn in observable_fns.items()
            }
            if cfg.observables.density:
                sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                observable_states['density'] = density_update(
                    subkeys, params, data, observable_states['density'])

            # 更新MCMC移动宽度
            mcmc_width, pmoves = mcmc.update_mcmc_width(
                t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

            # 检查NaN并处理
            if cfg.debug.check_nan:
                tree = {'params': params, 'loss': loss}
                if cfg.optim.optimizer != 'none':
                    tree['optim'] = opt_state
                try:
                    chex.assert_tree_all_finite(tree)
                    num_resets = 0
                except AssertionError as e:
                    if cfg.optim.reset_if_nan:
                        num_resets += 1
                        if num_resets > 100:
                            raise e
                    else:
                        raise e

            # 日志记录
            if t % cfg.log.stats_frequency == 0:
                logging_str = ('Step %05d: '
                               '%03.4f s, %03.4f E_h, %03.4f E_h^2, exp. variance=%03.4f E_h^2, pmove=%0.2f')
                logging_args = t, time.time()-start_time, loss, aux_data.variance[0].sum(), weighted_stats.variance, pmove
                writer_kwargs = dict(
                    step =  t,
                    time = time.time()-start_time,
                    energy = np.asarray(loss),
                    var = np.asarray(aux_data.variance[0].sum()),
                    ewmean = np.asarray(weighted_stats.mean),
                    ewvar = np.asarray(weighted_stats.variance),
                    pmove = np.asarray(pmove),
                    grad_norm = stats['grad_norm'][0],
                    learning_rate = stats['learning_rate'][0],
                    momentum = stats['momentum'][0],
                    param_norm = stats['param_norm'][0],
                    precon_grad_norm = stats['precon_grad_norm'][0],
                    update_norm = stats['update_norm'][0],
                )
                start_time = time.time()
                for key in observable_data:
                    obs_data = observable_data[key]
                    if cfg.system.states:
                        obs_data = np.trace(obs_data, axis1=-1, axis2=-2)
                    if key == 'dipole':
                        writer_kwargs['mu_x'] = obs_data[0]
                        writer_kwargs['mu_y'] = obs_data[1]
                        writer_kwargs['mu_z'] = obs_data[2]
                    elif key == 'density':
                        pass
                    elif key == 's2':
                        writer_kwargs[key] = obs_data
                        logging_str += ', <S^2>=%03.4f'
                        logging_args += obs_data,
                logging.info(logging_str, *logging_args)
                if t>0 or os.path.exists(out_file_name):
                    df = pd.DataFrame(writer_kwargs, index=[0])
                    df.to_csv(out_file_name, mode='a', header=False)
                elif t == 0:
                    df = pd.DataFrame(writer_kwargs, index=[0])
                    df.to_csv(out_file_name, header=True, mode = 'w')

            # 处理过大的观测量数据，直接保存为.npy文件
            if cfg.system.states:
                energy_matrix = aux_data.local_energy_mat
                energy_matrixs = energy_matrix.reshape((cfg.batch_size,)+energy_matrix.shape[2:])
                energy_matrixs_clip = clip_local_energys(energy_matrixs)
                energy_matrix = np.nanmean(energy_matrixs, axis=0)
                energy_matrix_std = np.nanstd(energy_matrixs, axis=0)
                energy_matrix_clip = np.nanmean(energy_matrixs_clip, axis=0)
                energy_matrix_clip_std = np.nanstd(energy_matrixs_clip, axis=0)
                np.save(energy_matrix_file, energy_matrix)
                np.save(energy_matrix_std_file, energy_matrix_std)
                np.save(energy_matrix_clip_file, energy_matrix_clip)
                np.save(energy_matrix_clip_std_file, energy_matrix_clip_std)
                if cfg.observables.s2:
                    np.save(s2_matrix_file, observable_data['s2'])
                if cfg.observables.dipole:
                    np.save(dipole_matrix_file, observable_data['dipole'])
            if cfg.observables.density:
                np.save(density_matrix_file, observable_data['density'])

            # 检查点保存
            if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
                checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
                time_of_last_ckpt = time.time()

        # 关闭日志
        if cfg.system.states:
            energy_matrix_file.close()
            if cfg.observables.s2:
                s2_matrix_file.close()
            if cfg.observables.dipole:
                dipole_matrix_file.close()
        if cfg.observables.density:
            density_matrix_file.close()