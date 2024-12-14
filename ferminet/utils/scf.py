# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interaction with Hartree-Fock solver in pyscf."""

# Abbreviations used:
# SCF: self-consistent field (method). Another name for Hartree-Fock
# HF: Hartree-Fock method.
# RHF: restricted Hartre-Fock. Require molecular orbital for the i-th alpha-spin
#   and i-th beta-spin electrons to have the same spatial component.
# ROHF: restricted open-shell Hartree-Fock. Same as RHF except allows the number
#   of alpha and beta electrons to differ.
# UHF: unrestricted Hartre-Fock. Permits breaking of spin symmetry and hence
#   alpha and beta electrons to have different spatial components.
# AO: Atomic orbital. Underlying basis set (typically Gaussian-type orbitals and
#   built into pyscf).
# MO: molecular orbitals/Hartree-Fock orbitals. Single-particle orbitals which
#   are solutions to the Hartree-Fock equations.


from typing import Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from ferminet.utils import elements
from ferminet.utils import gto
from ferminet.utils import system
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pyscf
import pyscf.gto
import pyscf.lib
import pyscf.scf


NDArray = Union[jnp.ndarray, np.ndarray]

# For excitations, tuple of (spin, orbital) pairs giving the orbitals which are
# swapped.
SpinOrbitalSwap = tuple[tuple[int, int], tuple[int, int]]


class Scf:
  """Helper class for running Hartree-Fock (self-consistent field) with pyscf.

  Attributes:
    molecule: list of system.Atom objects giving the atoms in the
      molecule and their positions.
    nelectrons: Tuple with number of alpha electrons and beta
      electrons.
    basis: Basis set to use, best specified with the relevant string
      for a built-in basis set in pyscf. A user-defined basis set can be used
      (advanced). See https://sunqm.github.io/pyscf/gto.html#input-basis for
        more details.
    pyscf_mol: the PySCF 'Molecule'. If this is passed to the init,
      the molecule, nelectrons, and basis will not be used, and the
      calculations will be performed on the existing pyscf_mol
    restricted: If true, use the restricted Hartree-Fock method, otherwise use
      the unrestricted Hartree-Fock method.
    mean_field: the actual UHF object.
    mo_coeff: The molecular orbital coefficients computed by Hartree-Fock.
    excitations: Stores a list of atomic orbitals to swap to construct excited
      states of the lowest energy.
  """

  def __init__(self,
               molecule: Sequence[system.Atom] | None = None,
               nelectrons: Tuple[int, int] | None = None,
               basis: str | None = 'cc-pVTZ',
               ecp: Mapping[str, str] | None = None,
               core_electrons: Mapping[str, int] | None = None,
               pyscf_mol: pyscf.gto.Mole | None = None,
               method: str = 'hf',
               restricted: bool = True):
    pyscf.lib.param.TMPDIR = None

    if pyscf_mol:
      self._mol = pyscf_mol
    else:
      # If not passed a pyscf molecule, create one
      if any(atom.atomic_number - atom.charge > 1.e-8
             for atom in molecule):
        logging.info(
            'Fractional nuclear charge detected. '
            'Running SCF on atoms with integer charge.'
        )
      ecp = ecp or {}
      core_electrons = core_electrons or {}

      nuclear_charge = 0
      for atom in molecule:
        nuclear_charge += atom.atomic_number
        if atom.symbol in core_electrons:
          nuclear_charge -= core_electrons[atom.symbol]
      charge = nuclear_charge - sum(nelectrons)
      self._mol = pyscf.gto.Mole(
          atom=[[atom.symbol, atom.coords] for atom in molecule],
          unit='bohr')
      self._mol.basis = basis
      self._mol.spin = nelectrons[0] - nelectrons[1]
      self._mol.charge = charge
      self._mol.ecp = ecp
      self._mol.build()
      if self._mol.nelectron != sum(nelectrons):
        raise RuntimeError('PySCF molecule not consistent with QMC molecule.')
      self._mol_jax = gto.Mol.from_pyscf_mol(self._mol)
    if restricted:
      self.mean_field = pyscf.scf.RHF(self._mol)
    else:
      self.mean_field = pyscf.scf.UHF(self._mol)

    # Create pure-JAX Mol object so that GTOs can be evaluated in traced
    # JAX functions
    self._mol_jax = gto.Mol.from_pyscf_mol(self._mol)
    self.restricted = restricted
    self.excitations = None
    self.method = method
    self.ci_vectors = None
    self.alpha_strings = None
    self.beta_strings = None
    self.ncore = None
    self.n_determinants = 16

  def run(self,
          dm0: np.ndarray | None = None,
          excitations: int = 0,
          excitation_type: str = 'ordered'):
    """Runs the Hartree-Fock calculation.

    Args:
      dm0: Optional density matrix to initialize the calculation.
      excitations: Stores a list of atomic orbitals to swap to construct excited
        states of the lowest energy.
      excitation_type: The way to construct different states for excited state
        pretraining. One of 'ordered' or 'random'. 'Ordered' tends to work
        better, but 'random' is necessary for some systems, especially double
        excitaitons.

    Returns:
      A pyscf scf object (i.e. pyscf.scf.rhf.RHF, pyscf.scf.uhf.UHF or
      pyscf.scf.rohf.ROHF depending on the spin and restricted settings).

    Raises:
      RuntimeError: If the number of electrons in the PySCF molecule is not
      consistent with self.nelectrons.
    """
    try:
      self.mean_field.kernel(dm0=dm0)
      if self.method == 'casci':
        self.initialize_casci(ncas=8, nelec=(3,3), nroots=excitations+1)
      elif self.method == 'casscf':
        self.initialize_casscf(ncas=8, nelec=(3,3), nroots=excitations+1, state_avg=True, fix_spin=None)

    except TypeError:
      logging.info('Mean-field solver does not support specifying an initial '
                   'density matrix.')
      # 1e solvers (e.g. uhf.HF1e) do not take any keyword arguments.
      self.mean_field.kernel()
    if excitations > 0:
      if excitation_type == 'ordered':
        self.excitations = get_ordered_excitations(
            self.mean_field, n=excitations, preserve_spin=True)
      elif excitation_type == 'random':
        nelec = self._mol.nelec
        ncore = 0  # number of core electrons (do not excite!)
        for atom in self._mol.atom:
          period = elements.SYMBOLS[atom[0]].period
          if period > 1 and atom[0] not in self._mol.ecp:
            ncore += elements.PERIODS[period-1][-1].atomic_number
        ncore //= 2  # core electrons are symmetric between alpha and beta spin
        assert excitations < np.prod([n-ncore+1 for n in nelec])
        nbasis = self._mol.nao
        self.excitations = [tuple(
            [np.concatenate(
                [np.concatenate([np.eye(ncore),
                                 np.zeros((ncore, n-ncore))], axis=1),
                 np.concatenate([np.zeros((n-ncore+1, ncore)),
                                 np.random.randn(n-ncore+1, n-ncore) /
                                 np.sqrt(n-ncore)], axis=1),
                 np.zeros((nbasis-n-1, n))], axis=0) for n in nelec])
                            for _ in range(excitations)]
      else:
        raise ValueError(f'Invalid excitation type: {excitation_type}')
    return self.mean_field

  def initialize_casci(self, ncas: int, nelec: Tuple[int, int], nroots: int = 1):
      """
      Initialize CASCI parameters.

      Args:
          ncas: Number of active space orbitals.
          nelec: Tuple of (alpha electrons, beta electrons).
          nroots: Number of states (including excited states) to compute.
      """
      from pyscf import gto, scf, mcscf
      from pyscf.fci import cistring

      mf = scf.RHF(self._mol).run()
      # 设置 CASCI 参数
      casci = mcscf.CASCI(mf, ncas, sum(nelec))
      casci.fcisolver.nroots = nroots
      casci.kernel()
      # Store CASCI results in class
      self.ci_vectors = casci.ci
      self.alpha_strings = cistring.gen_strings4orblist(range(ncas), nelec[0])
      self.beta_strings = cistring.gen_strings4orblist(range(ncas), nelec[1])
      self.ncore = casci.ncore

  def initialize_casscf(self, ncas: int, nelec: Tuple[int, int], nroots: int = 1, state_avg: bool = True, fix_spin: Optional[float] = None):
      """
      Initialize CASSCF parameters.

      Args:
          ncas: Number of active space orbitals.
          nelec: Tuple of (alpha electrons, beta electrons).
          nroots: Number of states (including excited states) to compute.
          state_avg: Whether to use state-averaged CASSCF.
          fix_spin: Target spin state (S^2 value).
      """
      from pyscf import scf, mcscf
      from pyscf.fci import cistring
      # 基于 RHF 初始化 CASSCF
      mf = scf.RHF(self._mol).run()
      casscf = mcscf.CASSCF(mf, ncas, sum(nelec))

      if nroots > 1:
          casscf.fcisolver.nroots = nroots
          if state_avg:
              casscf.state_average_([1.0 / nroots] * nroots)

      if fix_spin is not None:
          casscf.fix_spin_(ss=fix_spin)

      # 运行 CASSCF
      casscf.kernel()

      # 存储结果
      self.casscf = casscf
      self.ci_vectors = casscf.ci
      self.alpha_strings = cistring.gen_strings4orblist(range(ncas), nelec[0])
      self.beta_strings = cistring.gen_strings4orblist(range(ncas), nelec[1])
      self.ncore = casscf.ncore

  @property
  def mo_coeff(self) -> Optional[np.ndarray]:
    if self.mean_field.mo_coeff is None:
      return None
    elif self.restricted:
      return np.asarray(self.mean_field.mo_coeff)[None]
    else:
      return np.asarray(self.mean_field.mo_coeff)

  @mo_coeff.setter
  def mo_coeff(self, mo_coeff):
    # pytype: disable=attribute-error
    if (self.mean_field is not None and
        self.mean_field.mo_coeff is not None and
        self.mean_field.mo_coeff.ndim != mo_coeff.ndim):
      raise ValueError('Attempting to override mo_coeffs with different rank. '
                       f'Got {mo_coeff.shape=}, have '
                       f'{self.mean_field.mo_coeff.shape=}')
    # pytype: enable=attribute-error
    self.mean_field.mo_coeff = mo_coeff

  def eval_mos(self, positions: NDArray) -> Tuple[NDArray, NDArray]:
    """Evaluates the Hartree-Fock single-particle orbitals at a set of points.

    Args:
      positions: numpy array of shape (N, 3) of the positions in space at which
        to evaluate the Hartree-Fock orbitals.

    Returns:
      Pair of numpy float64 arrays of shape (N, M) (deriv=False) or (4, N, M)
      (deriv=True), where 2M is the number of Hartree-Fock orbitals. The (i-th,
      j-th) element in the first (second) array gives the value of the j-th
      alpha (beta) Hartree-Fock orbital at the i-th electron position in
      positions. For restricted (RHF, ROHF) calculations, the two arrays will be
      identical.
      If deriv=True, the first index contains [value, x derivative, y
      derivative, z derivative].

    Raises:
      RuntimeError: If Hartree-Fock calculation has not been performed using
        `run`.
      NotImplementedError: If Hartree-Fock calculation used Cartesian
        Gaussian-type orbitals as the underlying basis set.
    """
    if self.mean_field is None:
      raise RuntimeError('Mean-field calculation has not been run.')
    if self.restricted:
      coeffs = (self.mean_field.mo_coeff,)
    else:
      coeffs = self.mean_field.mo_coeff
    # Assumes self._mol.cart (use of Cartesian Gaussian-type orbitals and
    # integrals) is False (default behaviour of pyscf).
    if self._mol.cart:
      raise NotImplementedError(
          'Evaluation of molecular orbitals using cartesian GTOs.')
    ao_values = self._mol_jax.eval_gto(positions)
    mo_values = tuple(jnp.matmul(ao_values, coeff) for coeff in coeffs)
    if self.restricted:
      # duplicate for beta electrons.
      mo_values *= 2
    return mo_values

  def eval_orbitals(self,
                   pos: NDArray,
                   nspins: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
    """Evaluates SCF orbitals at a set of positions.

    Args:
      pos: an array of electron positions to evaluate the orbitals at, of shape
        (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
        number of electrons and the spin up electrons are ordered before the
        spin down electrons.
      nspins: tuple with number of spin up and spin down electrons.
    Returns:
      tuple with matrices of orbitals for spin up and spin down electrons, with
      the same leading dimensions as in pos.
    """
    if self.method == 'hf':
      return self.eval_hf_orbitals(pos, nspins)
    else:
      return self.eval_cas_orbitals(pos, nspins)


  def eval_hf_orbitals(self,
                    pos: NDArray,
                    nspins: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
    """Evaluates SCF orbitals at a set of positions.

    Args:
      pos: an array of electron positions to evaluate the orbitals at, of shape
        (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
        number of electrons and the spin up electrons are ordered before the
        spin down electrons.
      nspins: tuple with number of spin up and spin down electrons.

    Returns:
      tuple with matrices of orbitals for spin up and spin down electrons, with
      the same leading dimensions as in pos.
    """
    if not isinstance(pos, np.ndarray):  # works even with JAX array
      try:
        pos = pos.copy()
      except AttributeError as exc:
        raise ValueError('Input must be either NumPy or JAX array.') from exc
    leading_dims = pos.shape[:-1]
    # split into separate electrons
    pos = jnp.reshape(pos, [-1, 3])  # (batch*nelec, 3)
    mos = self.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
    # Reshape into (batch, nelec, nbasis) for each spin channel.
    mos = [jnp.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
    # Return (using Aufbau principle) the matrices for the occupied alpha and
    # beta orbitals. Number of alpha electrons given by nspins[0].
    alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
    beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
    if self.excitations is not None:
      # Some indexing gymnastics to get out excited states as well.
      # Excited states are given along the second index, after the batch.
      alpha_spins = [alpha_spin]
      beta_spins = [beta_spin]
      # TODO(pfau, jamessspencer): More jaxlike to vmap over excitations.
      for excitation in self.excitations:
        alpha_excited = alpha_spin.copy()
        beta_excited = beta_spin.copy()
        if isinstance(excitation[0], float):
          # excitation_type == 'ordered'
          for occ_index, unocc_index in excitation[2]:
            spin_occ, i_occ = occ_index
            spin_unocc, i_unocc = unocc_index
            if spin_occ == 0:
              alpha_excited = alpha_excited.at[..., i_occ].set(
                  mos[spin_unocc][..., :nspins[0], i_unocc])
            elif spin_occ == 1:
              beta_excited = beta_excited.at[..., i_occ].set(
                  mos[spin_unocc][..., nspins[0]:, i_unocc])
            else:
              raise ValueError(f'Invalid {spin_occ=}')
        elif isinstance(excitation[0], NDArray):
          # excitation_type == 'random'
          alpha_excited = jnp.dot(mos[0][:, :nspins[0]], excitation[0])
          beta_excited = jnp.dot(mos[1][:, nspins[0]:], excitation[1])
        else:
          raise ValueError(f'Unexpected excitation type: {type(excitation[0])}')
        alpha_spins.append(alpha_excited)
        beta_spins.append(beta_excited)
      alpha_spin = jnp.stack(alpha_spins, axis=-3)
      beta_spin = jnp.stack(beta_spins, axis=-3)
    return alpha_spin[:,:,None,...], beta_spin[:,:,None,...] # batch, n_states, n_states, n_determinants, n_alpha, n_alpha

  def eval_cas_orbitals(self,
                          pos: NDArray,
                          nspins: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
      """
      Evaluate CASCI orbitals for the largest n_determinants contributions.

      Args:
          pos: Electron positions, shape (N_sample, n_electrons * 3).
          nspins: Tuple of number of spin-up and spin-down electrons.

      Returns:
          Tuple of:
            - Alpha-spin orbitals of shape (N_sample, N_states, N_determinants, N_alpha, N_alpha)
            - Beta-spin orbitals of shape (N_sample, N_states, N_determinants, N_beta, N_beta)
      """
      # Ensure ci_vectors, alpha_strings, and beta_strings are initialized
      ci_vectors = self.ci_vectors
      alpha_strings = self.alpha_strings
      beta_strings = self.beta_strings

      if not isinstance(pos, np.ndarray):  # Ensure `pos` is a NumPy array
          try:
              pos = pos.copy()
          except AttributeError as exc:
              raise ValueError('Input must be either NumPy or JAX array.') from exc

      # Prepare dimensions
      leading_dims = pos.shape[:-1]  # Batch dimensions
      n_samples = pos.shape[0]  # Number of samples
      n_states = len(ci_vectors)  # Number of CASCI states
      n_alpha, n_beta = nspins  # Spin configuration

      # Reshape positions to separate electrons
      pos = jnp.reshape(pos, [-1, 3])  # (N_sample * n_electrons, 3)
      

      # Evaluate molecular orbitals
      mos = self.eval_mos(pos)  # (N_sample * n_electrons, nbasis)
      mos = [jnp.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
      # Include core orbitals in the occupied indices
      n_core = self.ncore  # Number of core orbitals
      core_indices = list(range(n_core))  # Core orbitals are always occupied

      # Initialize output for all states
      alpha_casci_orbitals = []
      beta_casci_orbitals = []

      # Loop over states 
      for state_idx in range(n_states):
          ci_vec = ci_vectors[state_idx]

          # Get top n_determinants indices based on |CI| values
          top_indices = np.argsort(-np.abs(ci_vec.flatten()))[:self.n_determinants]

          state_alpha_orbitals = []
          state_beta_orbitals = []

          # Loop over top determinants
          for idx in top_indices:
              alpha_idx, beta_idx = np.unravel_index(idx, ci_vec.shape)
              # Decode occupation for active space
              active_alpha_occ = [
                  i for i in range(len(alpha_strings))
                  if (alpha_strings[alpha_idx] & (1 << i))
              ]
              active_beta_occ = [
                  i for i in range(len(beta_strings))
                  if (beta_strings[beta_idx] & (1 << i))
              ]

              # Combine core and active orbitals
              alpha_occ = core_indices + [n_core + i for i in active_alpha_occ]
              beta_occ = core_indices + [n_core + i for i in active_beta_occ]
              print(f"state_idx:{state_idx}", alpha_occ, beta_occ, ci_vec[alpha_idx, beta_idx])
              # Extract molecular orbitals for spin channels
              alpha_orbitals = mos[0][..., :n_alpha, alpha_occ]*ci_vec[alpha_idx, beta_idx]
              beta_orbitals = mos[1][..., n_alpha:, beta_occ]*ci_vec[alpha_idx, beta_idx]

              # Append determinants for current state
              state_alpha_orbitals.append(alpha_orbitals)
              state_beta_orbitals.append(beta_orbitals)

          # Stack determinants for this state
          alpha_casci_orbitals.append(jnp.stack(state_alpha_orbitals, axis=-3))
          beta_casci_orbitals.append(jnp.stack(state_beta_orbitals, axis=-3))

      # Stack all states into separate arrays for alpha and beta spins
      alpha_casci_orbitals = jnp.stack(alpha_casci_orbitals, axis=-4)
      beta_casci_orbitals = jnp.stack(beta_casci_orbitals, axis=-4)
      return alpha_casci_orbitals, beta_casci_orbitals

  def eval_slater(self,
                  pos: Union[jnp.ndarray, np.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates the Slater determinant.

    Args:
      pos: an array of electron positions to evaluate the orbitals at.
      nspins: tuple with number of spin up and spin down electrons.

    Returns:
      tuple with sign and log absolute value of Slater determinant.
    """
    matrices = self.eval_orbitals(pos, nspins)
    slogdets = [jnp.linalg.slogdet(elem) for elem in matrices]
    sign_alpha, sign_beta = [elem[0] for elem in slogdets]
    log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
    log_abs_slater_determinant = log_abs_wf_alpha + log_abs_wf_beta
    sign = sign_alpha * sign_beta
    return sign, log_abs_slater_determinant


# pylint: disable=protected-access
def scf_flatten(scf: Scf):
  # `children` are fields which require an extra leading dimension with pmap
  children = ()
  # `aux_data` are any fields which are required to reconstruct the original
  # object, but would not actually end up as data in the compiled graph.
  aux_data = (scf.mo_coeff,
              scf._mol_jax._spec,
              scf._mol,
              scf.restricted,
              scf.n_determinants,
              scf.method,
              scf.ci_vectors,
              scf.alpha_strings,
              scf.beta_strings,
              scf.ncore,
              scf.excitations)
  return children, aux_data


def scf_unflatten(aux_data, children) -> Scf:
  assert not children  # children should be empty.
  mo_coeff, spec, mol, restricted, n_determinants, method, ci_vectors, alpha_strings, beta_strings, ncore, excitations = aux_data
  scf = Scf(pyscf_mol=mol.copy(), restricted=restricted)
  scf.mo_coeff = mo_coeff
  scf._mol_jax._spec = spec
  scf.excitations = excitations
  scf.n_determinants = n_determinants
  scf.method = method
  scf.ci_vectors = ci_vectors
  scf.alpha_strings = alpha_strings
  scf.beta_strings = beta_strings
  scf.ncore = ncore
  return scf
# pylint: enable=protected-access


jax.tree_util.register_pytree_node(Scf, scf_flatten, scf_unflatten)


def get_ordered_excitations(
    mean_field: ...,
    n: int = 10,
    preserve_spin: bool = False
) -> list[tuple[float, int, list[SpinOrbitalSwap]]]:
  """Compute energies of lowest n single/double excitations from HF result.

  Args:
     mean_field: scf mean_field returned by pyscf.
     n: Maximum number of excitations to return.
     preserve_spin: If true, return only excitations which preserve spin.

  Returns:
    List of at most n excitations, where each excitation is a tuple of energy
    difference of the HF eigenvalues of the occupied orbitals, change in spin,
    followed by an arbitrary number of tuples containing pairs of occupied /
    unoccupied indices, each of which is a tuple of (spin, orbital), giving the
    orbitals which are swapped.
  """
  occ, energy = mean_field.mo_occ, mean_field.mo_energy
  nocc = [int(occ[spin].sum()) for spin in range(2)]
  norb = [len(occ[spin]) for spin in range(2)]
  # A list of tuples containing energy difference, change in spin, followed by
  # an arbitrary number of tuples containing pairs of occupied / unoccupied
  # indices, each of which is a tuple of (spin, orbital), giving the orbitals
  # which are swapped.
  res = []

  # Single excitations

  for spin_occ in range(2):
    for i_occ in range(nocc[spin_occ]):
      for spin_unocc in range(2):
        for i_unocc in range(nocc[spin_unocc], norb[spin_unocc]):
          delta_e = (energy[spin_unocc][i_unocc] - energy[spin_occ][i_occ])
          res.append((delta_e, spin_occ - spin_unocc,
                      [((spin_occ, i_occ), (spin_unocc, i_unocc))]))

  # Double excitations
  for spin_occ1 in range(2):
    for i_occ1 in range(nocc[spin_occ1]):
      for spin_unocc1 in range(2):
        for i_unocc1 in range(nocc[spin_unocc1], norb[spin_unocc1]):
          for spin_occ2 in range(2):
            for i_occ2 in range(nocc[spin_occ2]):
              for spin_unocc2 in range(2):
                for i_unocc2 in range(nocc[spin_unocc2], norb[spin_unocc2]):
                  occ_index1 = spin_occ1, i_occ1
                  occ_index2 = spin_occ2, i_occ2
                  unocc_index1 = spin_unocc1, i_unocc1
                  unocc_index2 = spin_unocc2, i_unocc2
                  if ((occ_index1 < occ_index2) and
                      (unocc_index1 < unocc_index2)):
                    delta_e = (
                        energy[spin_unocc1][i_unocc1] +
                        energy[spin_unocc2][i_unocc2] -
                        energy[spin_occ1][i_occ1] - energy[spin_occ2][i_occ2])
                    res.append(
                        (delta_e,
                         spin_occ1 + spin_occ2 - spin_unocc1 - spin_unocc2, [
                             (occ_index1, unocc_index1),
                             (occ_index2, unocc_index2)
                         ]))

  if preserve_spin:
    res = [x for x in res if x[1] == 0]
  if len(res) < n:
    raise ValueError('Insufficient single and double excitations. '
                     f'Want {n}, have {len(res)}. Try a larger basis set?')
  # Take the first n excitations.
  # Note that this has a heavy bias towards single excitations.
  return sorted(res, key=lambda x: x[0])[:n]
