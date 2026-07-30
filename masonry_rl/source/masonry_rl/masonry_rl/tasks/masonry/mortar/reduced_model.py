"""Track A: reduced-order mortar model used for *training*.

The mortar bed is represented as a thickness field rather than particles, so the
whole thing is a handful of ``torch`` ops and runs at any number of parallel
environments. It deliberately does not try to be physically accurate; it
preserves the causal structure the policy has to learn:

* extrude nothing -> the brick does not bond (no coverage, no cure)
* press too little -> the joint stays too thick
* press too hard  -> mortar squeezes out (wasted, penalised)
* disturb before it cures -> the brick moves

The high fidelity PBD particle mortar (Track B) lives in ``pbd_demo.py`` and is
used only for rendering the final demo. Do not import it here.

Like :mod:`wall_planner` this module has no Isaac Sim dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["MortarCfg", "MortarField"]


@dataclass
class MortarCfg:
    num_cells: int = 8
    """Sample cells per slot, spread along the brick's length."""

    nominal_joint: float = 0.010
    """Target joint thickness after seating [m]."""
    min_joint: float = 0.008
    max_joint: float = 0.012
    """Acceptance band for the ``joint_thickness`` reward term [m]."""

    target_deposit: float = 0.012
    """Bed thickness the policy should extrude before seating [m].

    Slightly above ``nominal_joint`` so that seating always compresses.
    """
    max_deposit: float = 0.030
    """Hard cap on bed thickness, prevents unbounded reward farming [m]."""

    extrusion_rate: float = 0.06
    """Bed growth at full extrusion command [m/s]."""
    nozzle_radius: float = 0.035
    """Cells within this planar distance of the nozzle receive mortar [m]."""

    coverage_threshold: float = 0.004
    """A cell counts as covered above this thickness [m]."""
    min_coverage: float = 0.75
    """Coverage required for a placement to be considered bonded."""

    cure_steps: int = 30
    """Consecutive stable steps before the joint is treated as cured."""

    squeeze_budget: float = 2.0e-5
    """Squeezed-out volume tolerated per slot before penalty [m^3]."""


class MortarField:
    """Batched mortar bed state: ``[num_envs, num_slots, num_cells]``.

    Every environment works on one slot at a time, but state is kept for all
    slots because previously laid joints determine the levelness of the courses
    above them.
    """

    def __init__(
        self,
        num_envs: int,
        num_slots: int,
        cfg: MortarCfg | None = None,
        cell_area: float = 0.190 * 0.090 / 8,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg or MortarCfg()
        self.num_envs = num_envs
        self.num_slots = num_slots
        self.cell_area = cell_area
        self.device = torch.device(device)

        shape = (num_envs, num_slots, self.cfg.num_cells)
        self.thickness = torch.zeros(shape, device=self.device)
        self.squeeze_out = torch.zeros(num_envs, num_slots, device=self.device)
        self.cure_counter = torch.zeros(num_envs, num_slots, device=self.device)
        self.cured = torch.zeros(num_envs, num_slots, dtype=torch.bool, device=self.device)

    # -- lifecycle ----------------------------------------------------------

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device).long()
        self.thickness[env_ids] = 0.0
        self.squeeze_out[env_ids] = 0.0
        self.cure_counter[env_ids] = 0.0
        self.cured[env_ids] = False

    # -- internal helpers ---------------------------------------------------

    def _gather(self, buf: torch.Tensor, slot_idx: torch.Tensor) -> torch.Tensor:
        """Select the active slot per environment from a ``[E, S, ...]`` buffer."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        return buf[env_ids, slot_idx.to(self.device).long()]

    def _scatter(self, buf: torch.Tensor, slot_idx: torch.Tensor, value: torch.Tensor) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        buf[env_ids, slot_idx.to(self.device).long()] = value

    # -- dynamics -----------------------------------------------------------

    def deposit(
        self,
        slot_idx: torch.Tensor,
        nozzle_pos: torch.Tensor,
        cell_pos: torch.Tensor,
        rate: torch.Tensor,
        dt: float,
    ) -> None:
        """Extrude mortar onto the active slot's bed.

        Args:
            slot_idx: ``[E]`` active slot per environment.
            nozzle_pos: ``[E, 3]`` nozzle tip position, world frame.
            cell_pos: ``[E, C, 3]`` cell centers, from
                :meth:`WallPlanner.slot_cell_positions`.
            rate: ``[E]`` extrusion command in ``[0, 1]``.
            dt: step duration [s].
        """
        cfg = self.cfg
        planar = cell_pos[..., :2] - nozzle_pos[:, None, :2]
        within = planar.norm(dim=-1) < cfg.nozzle_radius  # [E, C]

        rate = rate.to(self.device).clamp(0.0, 1.0)[:, None]
        added = within.to(self.thickness.dtype) * rate * cfg.extrusion_rate * dt

        current = self._gather(self.thickness, slot_idx)
        self._scatter(
            self.thickness, slot_idx, (current + added).clamp(max=cfg.max_deposit)
        )

    def press(
        self,
        slot_idx: torch.Tensor,
        brick_bottom_z: torch.Tensor,
        bed_base_z: torch.Tensor,
    ) -> torch.Tensor:
        """Seat a brick: compress the bed to the available gap.

        Mortar displaced by the brick is accumulated as squeeze-out.

        Returns:
            ``[E]`` the resulting joint thickness (the gap), clamped at zero.
        """
        gap = (brick_bottom_z - bed_base_z).clamp(min=0.0).to(self.device)

        before = self._gather(self.thickness, slot_idx)
        after = torch.minimum(before, gap[:, None])
        self._scatter(self.thickness, slot_idx, after)

        displaced = (before - after).sum(dim=-1) * self.cell_area
        squeezed = self._gather(self.squeeze_out, slot_idx)
        self._scatter(self.squeeze_out, slot_idx, squeezed + displaced)
        return gap

    def tick_cure(self, slot_idx: torch.Tensor, stable: torch.Tensor) -> torch.Tensor:
        """Advance the cure timer for the active slot.

        Args:
            slot_idx: ``[E]`` active slot per environment.
            stable: ``[E]`` bool, whether the brick held still this step.

        Returns:
            ``[E]`` bool, whether the joint is now cured.
        """
        stable = stable.to(self.device)
        bonded = self.coverage(slot_idx) >= self.cfg.min_coverage

        # The timer only runs while the joint is BOTH bonded and undisturbed.
        # Gating on `stable` alone would let a policy hold still over a dry bed,
        # bank the timer, then extrude at the last moment and cure instantly.
        counter = self._gather(self.cure_counter, slot_idx)
        counter = torch.where(stable & bonded, counter + 1.0, torch.zeros_like(counter))
        self._scatter(self.cure_counter, slot_idx, counter)

        prev = self._gather(self.cured, slot_idx)
        now = prev | (counter >= self.cfg.cure_steps)
        self._scatter(self.cured, slot_idx, now)
        return now

    # -- observations / reward signals --------------------------------------

    def coverage(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """Fraction of cells carrying mortar, ``[E]`` in ``[0, 1]``."""
        th = self._gather(self.thickness, slot_idx)
        return (th > self.cfg.coverage_threshold).to(th.dtype).mean(dim=-1)

    def mean_thickness(self, slot_idx: torch.Tensor) -> torch.Tensor:
        return self._gather(self.thickness, slot_idx).mean(dim=-1)

    def bed_quality(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """Reward signal for the ``mortar_bed`` term, ``[E]`` in ``[0, 1]``.

        Coverage scaled by how close the mean bed thickness is to
        ``target_deposit``; over-extrusion decays the score rather than
        saturating it.
        """
        cfg = self.cfg
        cover = self.coverage(slot_idx)
        err = (self.mean_thickness(slot_idx) - cfg.target_deposit).abs()
        return cover * torch.exp(-err / max(cfg.target_deposit, 1e-6))

    def joint_ok(self, joint_thickness: torch.Tensor) -> torch.Tensor:
        """``[E]`` bool: is the seated joint inside the acceptance band?"""
        cfg = self.cfg
        return (joint_thickness >= cfg.min_joint) & (joint_thickness <= cfg.max_joint)

    def squeeze_penalty(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """Squeezed-out volume beyond the budget, ``[E]`` >= 0."""
        excess = self._gather(self.squeeze_out, slot_idx) - self.cfg.squeeze_budget
        return excess.clamp(min=0.0)

    def observation(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """Per-slot mortar observation, ``[E, num_cells + 1]``.

        The cell thicknesses plus coverage; the extrusion command itself is part
        of the action history and is appended by the observation manager.
        """
        th = self._gather(self.thickness, slot_idx)
        return torch.cat([th, self.coverage(slot_idx)[:, None]], dim=-1)
