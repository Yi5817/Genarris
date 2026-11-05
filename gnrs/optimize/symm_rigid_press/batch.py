"""
Batched torch backend of the symmetry-preserving rigid press.

Many crystals are optimized at once on one GPU. The
energy is the one of ContactEnergy, written for a batch of padded tensors
and differentiated with autograd; every crystal runs its own BFGS iteration.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
import torch
from ase import Atoms

from gnrs.optimize.symm_rigid_press.energy import CENTRE_TOL, MAX_CELL_SUM, WALL_BAND
from gnrs.optimize.symm_rigid_press.optimizer import RigidPressSymm

# Lattice parameters of a padded state (the triclinic count)
N_LAT = 6
# Line search: sufficient decrease constant, maximum number of halvings of the
# trial step and halvings tested at once after the first one
ARMIJO = 1e-4
MAX_HALVINGS = 30
CANDIDATES = 8


def skew(v: torch.Tensor) -> torch.Tensor:
    """
    Skew-symmetric matrices [v]x of vectors (..., 3).
    """
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape, 3)


class PairContacts(torch.autograd.Function):
    """
    Contact energy (T,) and smallest d - r (T,) of the atom pairs of T molecule
    pairs: molecule 1 (T, n, 3) against molecule 2 (T, n, 3) displaced by
    shifts (T, 3), the kernel of energy.pair_energy with its analytic
    gradient; the backward keeps one (T, n, n) tensor instead of autograd's
    dozen.
    """

    @staticmethod
    def forward(
        ctx,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
        shifts: torch.Tensor,
        valid: torch.Tensor,
        radius: torch.Tensor,
        D: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diff = pos1[:, :, None] - (pos2 + shifts[:, None])[:, None]
        diff = torch.where(valid[..., None], diff, 1.0)  # padded atoms: finite distance
        dist = torch.linalg.norm(diff, dim=-1)
        gap = dist - radius
        active = (dist < D) & valid
        # Finite wall: below r + WALL_BAND the kernel continues linearly
        gap_safe = gap.clamp(min=WALL_BAND)
        deriv = torch.where(active, -weight * (D - radius) / gap_safe**2, 0.0)
        energy = torch.where(
            gap < WALL_BAND,
            weight * (D - radius - WALL_BAND) / WALL_BAND + deriv * (gap - WALL_BAND),
            weight * (D - dist) / gap_safe,
        )
        # dE/dd / d, so that dE/dpos = coeff * diff in backward
        ctx.save_for_backward(pos1, pos2, shifts, valid, deriv / dist)
        return (
            torch.where(active, energy, 0.0).sum((1, 2)),
            torch.where(valid, gap, torch.inf).amin((1, 2)),
        )

    @staticmethod
    def backward(ctx, grad_energy: torch.Tensor, _: torch.Tensor) -> tuple:
        pos1, pos2, shifts, valid, coeff = ctx.saved_tensors
        diff = pos1[:, :, None] - (pos2 + shifts[:, None])[:, None]
        diff = torch.where(valid[..., None], diff, 0.0)
        grad = (grad_energy[:, None, None] * coeff)[..., None] * diff
        return grad.sum(2), -grad.sum(1), -grad.sum((1, 2)), None, None, None, None


def bfgs_update(H: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    BFGS update of the inverse Hessians (B, S, S) from the steps s and the
    gradient changes y; a crystal whose curvature s . y is not positive keeps
    its matrix.
    """
    sy = (s * y).sum(-1)
    # rho = 0 leaves H unchanged (Nocedal & Wright eq. 6.17 with rho = 1 / s.y)
    rho = torch.where(sy > 1e-10, 1 / sy, 0.0)[:, None, None]
    Hy = H @ y[..., None]
    s = s[..., None]
    yHy = (y[..., None] * Hy).sum(-2, keepdim=True)
    return H - rho * (Hy @ s.mT + s @ Hy.mT) + (rho**2 * yHy + rho) * (s @ s.mT)


class RigidPressBatch:
    """
    Optimize a batch of crystals at once with the energy of RigidPressSymm.

    Every crystal gets a padded state: 6 lattice parameters, then 3 centre
    coordinates and 3 rotation angles per molecule of the asymmetric unit.
    Padded entries have zero basis vectors, so their gradient vanishes. The
    state layout is [6 lattice | M x 3 centres | M x 3 angles] (`slots`).
    Molecules of the cell are indexed z * M + j (copy z, molecule j).
    """

    def __init__(self, opts: list[RigidPressSymm], device: str = "cuda") -> None:
        """
        Args:
            opts: One prepared optimizer per crystal (symmetry already analysed)
            device: torch device of the batch
        """
        self.opts = opts
        self.device = device
        coords = [o.coords for o in opts]
        B = len(opts)
        M = self.M = max(len(c.sites) for c in coords)
        Z = self.Z = max(c.z for c in coords)
        n = self.n = max(m.stop - m.start for c in coords for m in c.asu_mol_atoms)

        # Padded per-crystal geometry: M molecules, Z copies, n atoms
        self.basis = torch.zeros(B, N_LAT, 3, 3, dtype=torch.float64)
        self.origin = torch.zeros(B, M, 3, dtype=torch.float64)
        self.directions = torch.zeros(B, M, 3, 3, dtype=torch.float64)
        self.axes = torch.zeros(B, M, 3, 3, dtype=torch.float64)
        self.ref_pos = torch.zeros(B, M, n, 3, dtype=torch.float64)
        self.atom_mask = torch.zeros(B, Z, M, n, dtype=torch.bool)
        self.asu_rot = torch.zeros(B, Z, 3, 3, dtype=torch.float64)
        self.asu_trans = torch.zeros(B, Z, 3, dtype=torch.float64)
        self.radius = torch.zeros(B, M, M, n, n, dtype=torch.float64)
        self.D = torch.zeros(B, Z * M, Z * M, dtype=torch.float64)
        self.weight = torch.tensor(
            [o.contacts.weight for o in opts], dtype=torch.float64
        )
        for i, (opt, c) in enumerate(zip(opts, coords)):
            self.basis[i, : len(c.basis)] = torch.from_numpy(c.basis)
            self.asu_rot[i, : c.z] = torch.from_numpy(c.asu_rot)
            self.asu_trans[i, : c.z] = torch.from_numpy(c.asu_trans)
            n_mols = len(c.sites)
            D = torch.zeros(M, M, dtype=torch.float64)
            D[:n_mols, :n_mols] = torch.from_numpy(
                opt.contacts.D_cell[:n_mols, :n_mols]
            )
            self.D[i] = D.repeat(Z, Z)  # indexed by cell molecule z * M + j
            for j, (site, mol) in enumerate(zip(c.sites, c.asu_mol_atoms)):
                n_j = mol.stop - mol.start
                self.origin[i, j] = torch.from_numpy(site.origin)
                # Centre parameters are in Angstrom: fold 1 / scale into the directions
                k = site.directions.shape[1]
                self.directions[i, j, :, :k] = torch.from_numpy(
                    site.directions / c.cog_scale[j]
                )
                self.axes[i, j, :, : site.axes.shape[1]] = torch.from_numpy(site.axes)
                self.ref_pos[i, j, :n_j] = torch.from_numpy(site.ref_pos)
                self.atom_mask[i, : c.z, j, :n_j] = True
                for j2, mol2 in enumerate(c.asu_mol_atoms):
                    n_j2 = mol2.stop - mol2.start
                    self.radius[i, j, j2, :n_j, :n_j2] = torch.from_numpy(
                        opt.radius[mol, mol2]
                    )
        self.D_max = self.D.amax((1, 2))
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))

    def slots(self, i: int) -> list[slice]:
        """
        Where the parts of the NumPy state of crystal i (lattice parameters,
        then centre and rotation parameters per molecule) sit in its padded state.
        """
        coords = self.opts[i].coords
        n_mols = len(coords.sites)
        # Centres of molecule j start at N_LAT + 3 j, angles at N_LAT + 3 (M + j)
        starts = [0] + [
            N_LAT + 3 * (j if j < n_mols else self.M + j - n_mols)
            for j in range(2 * n_mols)
        ]
        sizes = [len(coords.basis), *coords.dofs]
        return [slice(s, s + size) for s, size in zip(starts, sizes)]

    def to_states(self, xtals: list[Atoms]) -> torch.Tensor:
        """
        Padded states of the crystals.
        """
        states = torch.zeros(len(xtals), N_LAT + 6 * self.M, dtype=torch.float64)
        for i, (opt, xtal) in enumerate(zip(self.opts, xtals)):
            state = torch.from_numpy(opt.coords.to_state(xtal))
            start = 0
            for slot in self.slots(i):
                size = slot.stop - slot.start
                states[i, slot] = state[start : start + size]
                start += size
        return states.to(self.device)

    def to_crystal(self, i: int, state: np.ndarray) -> Atoms:
        """
        Crystal i from its padded state.
        """
        return self.opts[i].coords.to_crystal(
            np.concatenate([state[slot] for slot in self.slots(i)])
        )

    def positions(
        self, state: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lattices and their inverses, Cartesian positions of the cell molecules
        (B, Z * M, n, 3) and which of these atoms are real, for crystals idx at
        the states given.
        """
        B, M, Z, n = len(state), self.M, self.Z, self.n
        lattice = torch.einsum("bkij,bk->bij", self.basis[idx], state[:, :N_LAT])
        centre, angles = state[:, N_LAT:].reshape(B, 2, M, 3).unbind(1)
        cog = self.origin[idx] + torch.einsum(
            "bmij,bmj->bmi", self.directions[idx], centre
        )
        rotation = torch.linalg.matrix_exp(
            skew(torch.einsum("bmij,bmj->bmi", self.axes[idx], angles))
        )
        offsets = self.ref_pos[idx] @ rotation.mT
        inv_lattice, info = torch.linalg.inv_ex(lattice)
        # A singular trial lattice gives NaN positions: no pairs, infinite energy
        inv_lattice = torch.where((info != 0)[:, None, None], torch.nan, inv_lattice)
        asu_frac = offsets @ inv_lattice[:, None] + cog[:, :, None]

        # Z copies of the asymmetric unit, each centred in the cell
        mask = self.atom_mask[idx]
        asu_mask = mask[:, 0, ..., None]
        asu_cog = (asu_frac * asu_mask).sum((1, 2)) / asu_mask.sum((1, 2))
        rot, trans = self.asu_rot[idx], self.asu_trans[idx]
        new_cog = torch.einsum("bzij,bj->bzi", rot, asu_cog) + trans
        frac = torch.einsum("bzij,bmnj->bzmni", rot, asu_frac)
        frac = frac + (trans - torch.floor(new_cog))[:, :, None, None]
        # Every molecule inside the cell, so that the image bound is complete
        mask = mask.reshape(B, Z * M, n)
        frac = frac.reshape(B, Z * M, n, 3)
        mol_cog = (frac * mask[..., None]).sum(2) / mask.sum(-1).clamp(min=1)[..., None]
        inside = (mol_cog > -CENTRE_TOL) & (mol_cog < 1 + CENTRE_TOL)
        frac = frac - torch.where(inside, 0.0, torch.floor(mol_cog))[:, :, None]
        return lattice, inv_lattice, frac @ lattice[:, None], mask

    def images(
        self, inv_lattice: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lattice displacements reaching every image within range: integer
        shifts (K, 3), which of them each crystal needs (B, K) and whether a
        cell is too small to sum over (B,).
        """
        # |n_k| <= D_max |column k of L^-1| covers every image within D_max
        bound = torch.linalg.norm(inv_lattice, dim=-2) * self.D_max[idx][:, None] + 1
        too_small = ~torch.isfinite(bound).all(-1) | (bound.prod(-1) > MAX_CELL_SUM)
        max_lat = torch.floor(torch.where(too_small[:, None], 0.0, bound)).long()
        grid = torch.cartesian_prod(
            *(torch.arange(-m, m + 1, device=idx.device) for m in max_lat.amax(0))
        ).reshape(-1, 3)
        in_range = (grid.abs()[None] <= max_lat[:, None]).all(-1)
        return grid, in_range, too_small

    def select_pairs(
        self,
        lattice: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        grid: torch.Tensor,
        in_range: torch.Tensor,
        too_small: torch.Tensor,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Indices (crystal, first molecule, second molecule, image) of the
        molecule pairs (first <= second) with centres closer than the pair's
        interaction distance; a molecule and its own image only for +L, not -L.
        """
        n_cell = pos.shape[1]
        mol_mask = atom_mask.any(-1)
        cog = (pos * atom_mask[..., None]).sum(2) / atom_mask.sum(-1).clamp(min=1)[
            ..., None
        ]
        shifts = grid.to(pos.dtype) @ lattice
        cog_dist = torch.linalg.norm(
            cog[:, :, None, None] - cog[:, None, :, None] - shifts[:, None, None],
            dim=-1,
        )
        # Upper triangle: each pair once; a self pair needs a nonzero shift
        upper = torch.ones(n_cell, n_cell, dtype=torch.bool, device=pos.device).triu()
        self_pair = torch.eye(n_cell, dtype=torch.bool, device=pos.device)
        select = (
            (cog_dist < self.D[idx][..., None])
            & in_range[:, None, None]
            & (upper & mol_mask[:, :, None] & mol_mask[:, None, :])[..., None]
            & (~self_pair[..., None] | grid.any(-1))
            & ~too_small[:, None, None, None]
        )
        return select.nonzero(as_tuple=True)

    def energy(
        self, state: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Volume plus contact energy of crystals idx at the states given
        (infinite if a cell is too small to sum over) and their smallest atom
        distance minus cutoff (negative: overlap).
        """
        lattice, inv_lattice, pos, atom_mask = self.positions(state, idx)
        M = self.M
        with torch.no_grad():
            grid, in_range, too_small = self.images(inv_lattice, idx)
            b, m1, m2, k = self.select_pairs(
                lattice, pos, atom_mask, grid, in_range, too_small, idx
            )
        D_cell = self.D[idx]

        # Contact energy of the atom pairs of the selected molecule pairs
        shifts = (grid[k].to(pos.dtype)[:, None] @ lattice[b])[:, 0]
        valid = atom_mask[b, m1][:, :, None] & atom_mask[b, m2][:, None]
        radius = self.radius[idx[b], m1 % M, m2 % M]
        D = D_cell[b, m1, m2][:, None, None]
        # +n and -n of a self pair are one image pair: half weight
        weight = (self.weight[idx[b]] * torch.where(m1 == m2, 0.5, 1.0))[:, None, None]
        pair, gap = PairContacts.apply(
            pos[b, m1], pos[b, m2], shifts, valid, radius, D, weight
        )
        energy = torch.zeros_like(too_small, dtype=pos.dtype).index_add(0, b, pair)
        total = torch.where(
            too_small, torch.inf, torch.linalg.det(lattice).abs() + energy
        )
        with torch.no_grad():
            min_gap = torch.full_like(total, torch.inf).scatter_reduce(
                0, b, gap, "amin"
            )
            min_gap = torch.where(too_small, -torch.inf, min_gap)
        return total, min_gap

    def evaluate(
        self, state: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Energy, its gradient on the state and the smallest gap of crystals idx.
        """
        state = state.detach().requires_grad_()
        energy, min_gap = self.energy(state, idx)
        (grad,) = torch.autograd.grad(energy.sum(), state)
        return energy.detach(), grad, min_gap

    def line_search(
        self,
        x: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        p: torch.Tensor,
        idx: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backtracking Armijo line search of crystals idx along p: the step
        length (0 where none was accepted) and whether one was.

        The first trial step is at most 1 Angstrom; a crystal that rejects it
        then tries the step halved up to MAX_HALVINGS times. After the first
        round the remaining crystals test up to CANDIDATES halvings in one
        call of at most batch_size rows.
        """
        slope = (g * p).sum(-1)
        # First trial: the full BFGS step, capped at 1 Angstrom
        t = (1 / p.norm(dim=-1)).clamp(max=1)
        accepted = torch.zeros_like(slope, dtype=torch.bool)
        live = torch.arange(len(x), device=x.device)
        n_cand, budget = 1, MAX_HALVINGS + 1
        while budget > 0 and live.numel():
            # Fewer live rows -> more halvings per call, so the batch stays full
            n_cand = min(n_cand, budget, max(1, batch_size // live.numel()))
            budget -= n_cand
            halve = 0.5 ** torch.arange(n_cand, device=x.device, dtype=x.dtype)
            cand = t[live, None] * halve
            with torch.no_grad():
                f_trial, _ = self.energy(
                    (x[live, None] + cand[..., None] * p[live, None]).flatten(0, 1),
                    idx[live].repeat_interleave(n_cand),
                )
            ok = (
                f_trial.view(-1, n_cand)
                <= f[live, None] + ARMIJO * cand * slope[live, None]
            )
            found = ok.any(1)
            # Largest candidate that passes, else continue halving from the last one
            first = ok.to(torch.int8).argmax(1)
            t[live] = torch.where(
                found, cand[torch.arange(len(live)), first], cand[:, -1] / 2
            )
            accepted[live[found]] = True
            live = live[~found]
            n_cand = CANDIDATES
        return torch.where(accepted, t, 0.0), accepted

    def run(
        self, xtals: list[Atoms], tol: float, maxiter: int, batch_size: int
    ) -> list[bool]:
        """
        Optimize the crystals in place with BFGS, one independent iteration
        per crystal; return whether each converged.

        At most batch_size crystals are evaluated at once: a crystal that stops
        leaves the active set and a waiting one takes its slot. Crystals whose
        atoms overlap at the start are left untouched. A crystal stops when its
        largest gradient component is below tol, after maxiter iterations, or
        when the line search fails.
        """
        x = self.to_states(xtals)
        N, S = x.shape
        f, gap = (
            torch.empty(N, dtype=x.dtype, device=x.device),
            torch.empty(N, dtype=x.dtype, device=x.device),
        )
        g = torch.empty_like(x)
        for idx in torch.arange(N, device=x.device).split(batch_size):
            f[idx], g[idx], gap[idx] = self.evaluate(x[idx], idx)
        optimized = gap >= 0  # overlapping at the start: left untouched
        converged = optimized & (g.abs().amax(-1) <= tol)
        done = ~optimized | converged
        H = torch.eye(S, dtype=x.dtype, device=x.device).repeat(N, 1, 1)
        iterations = torch.zeros(N, dtype=torch.long, device=x.device)

        # Continuous batching: stopped crystals leave, waiting ones fill the slots
        while (idx := (~done).nonzero()[:batch_size, 0]).numel():
            p = -(H[idx] @ g[idx][..., None])[..., 0]
            t, accepted = self.line_search(x[idx], f[idx], g[idx], p, idx, batch_size)
            step = t[:, None] * p
            f_new, g_new, gap_new = self.evaluate(x[idx] + step, idx)
            H[idx] = bfgs_update(H[idx], step, g_new - g[idx])
            x[idx] += step
            f[idx], g[idx], gap[idx] = f_new, g_new, gap_new
            iterations[idx] += 1
            converged[idx] = accepted & (g_new.abs().amax(-1) <= tol)
            done[idx] = converged[idx] | ~accepted | (iterations[idx] >= maxiter)

        x = x.cpu().numpy()
        for i, (xtal, was_optimized) in enumerate(zip(xtals, optimized.tolist())):
            if was_optimized:
                xtal_opt = self.to_crystal(i, x[i])
                xtal.set_cell(xtal_opt.cell)
                xtal.set_positions(xtal_opt.positions)
        return (converged & (gap >= 0)).tolist()
