"""The learning-rate sweep rules of plan §7.4: the grid, the edge rule and the tie rule.

Written by the orchestrator before the implementation (TASKS.md T-33, O+W). A
worker implements `analysis/lr_sweep.py` against this file and may not change it.

Plan §7.4, quoted:

1. MHA first, 5 points: 3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3, at the 100M budget, one seed.
2. Every other variant and both ladder sizes, 3 points: ½×, 1× and 2× MHA's winner.
3. Edge rule: if a winner is at the edge of its grid, add one point beyond it.
   Repeat until it isn't.
4. Tie rule: if a variant's best LR beats MHA's LR by less than σ_init at 100M
   from Arm N, use MHA's LR. A single-seed sweep otherwise selects on noise.
5. Record everything, including divergences.

Readings fixed here (decisions for Phil to confirm, logged in docs/decisions.md):

- The rules apply in the plan's order: the edge rule must be satisfied before
  the tie rule is read.
- A diverged point counts as a tested grid point that can never win. A winner
  next to a diverged edge point is therefore not at the edge.
- "Beats by less than σ_init" is strict: a gap equal to σ_init keeps the tuned rate.
- The point beyond an edge continues the factor-2 grid.

Every number in this file is made up.
"""

from __future__ import annotations

import copy

import pytest

from analysis import lr_sweep as sw
from src.utils import manifest as mf
from tests.fixtures.fake_manifests import fake_manifest

GRID = (3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3)
SWEEP_SEED = 21  # any seed; the module takes it as an argument


def pts(losses: dict) -> list:
    """Points from {lr: loss or None}; None is a diverged run."""
    return [
        sw.Point(lr=lr, loss=loss, diverged=loss is None, run=f"run-lr{mf.format_lr(lr)}", reason=None if loss is not None else "made up")
        for lr, loss in losses.items()
    ]


def lrs(choice) -> list[float]:
    return [p.lr for p in choice.points]


# --- The grids ------------------------------------------------------------------------------


def test_the_reference_grid_is_the_plans():
    assert sw.MHA_GRID == GRID


def test_a_variant_grid_is_half_one_and_two_times_the_reference_rate():
    assert sw.variant_grid(1.2e-3) == (6e-4, 1.2e-3, 2.4e-3)
    assert [mf.format_lr(x) for x in sw.variant_grid(2.4e-3)] == ["1.2e3", "2.4e3", "4.8e3"]


# --- MHA: the winner and the edge rule --------------------------------------------------------


def test_an_interior_winner_is_chosen():
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.52, 4.8e-3: 3.70}))
    assert c.status == sw.CHOSEN
    assert c.lr == 1.2e-3 and c.best_lr == 1.2e-3
    assert c.source_run == "run-lr1.2e3"
    assert c.next_lr is None and c.missing == ()
    assert lrs(c) == sorted(GRID)


def test_a_winner_at_the_top_edge_asks_for_the_next_point_up():
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.48, 4.8e-3: 3.45}))
    assert c.status == sw.EXTEND
    assert c.lr is None and c.source_run is None
    assert c.best_lr == 4.8e-3
    assert mf.format_lr(c.next_lr) == "9.6e3"


def test_a_winner_at_the_bottom_edge_asks_for_the_next_point_down():
    c = sw.choose_reference(pts({3e-4: 3.40, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.52, 4.8e-3: 3.70}))
    assert c.status == sw.EXTEND
    assert c.best_lr == 3e-4
    assert mf.format_lr(c.next_lr) == "1.5e4"


def test_the_edge_rule_repeats_until_the_winner_is_interior():
    losses = {3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.48, 4.8e-3: 3.45, 9.6e-3: 3.40}
    again = sw.choose_reference(pts(losses))
    assert again.status == sw.EXTEND and mf.format_lr(again.next_lr) == "1.92e2"
    losses[1.92e-2] = 3.90
    done = sw.choose_reference(pts(losses))
    assert done.status == sw.CHOSEN and done.lr == 9.6e-3


def test_a_missing_grid_point_makes_the_sweep_incomplete():
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.55, 2.4e-3: 3.52, 4.8e-3: 3.70}))
    assert c.status == sw.INCOMPLETE
    assert c.lr is None and c.next_lr is None
    assert c.missing == (1.2e-3,)


def test_a_gap_inside_an_extended_grid_is_incomplete_too():
    # 9.6e-3 was added by the edge rule, but 4.8e-3 is gone: the grid must be contiguous.
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.48, 9.6e-3: 3.47}))
    assert c.status == sw.INCOMPLETE
    assert c.missing == (4.8e-3,)


def test_a_diverged_top_point_bounds_the_grid():
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.45, 4.8e-3: None}))
    assert c.status == sw.CHOSEN
    assert c.lr == 2.4e-3
    diverged = [p for p in c.points if p.diverged]
    assert [p.lr for p in diverged] == [4.8e-3] and diverged[0].loss is None


def test_a_sweep_in_which_everything_diverged_is_an_error():
    with pytest.raises(sw.SweepError):
        sw.choose_reference(pts({lr: None for lr in GRID}))


def test_two_runs_at_one_rate_are_refused():
    doubled = pts({lr: 3.5 for lr in GRID}) + pts({1.2e-3: 3.4})
    with pytest.raises(sw.SweepError):
        sw.choose_reference(doubled)


def test_exactly_equal_losses_take_the_lower_rate():
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.50, 1.2e-3: 3.50, 2.4e-3: 3.52, 4.8e-3: 3.70}))
    assert c.status == sw.CHOSEN and c.lr == 6e-4


# --- A variant: the edge rule, then the tie rule ----------------------------------------------


REF = 1.2e-3
SIGMA = 0.004


def test_a_variant_whose_best_rate_is_the_reference_rate_uses_it():
    c = sw.choose_variant(pts({6e-4: 3.56, 1.2e-3: 3.50, 2.4e-3: 3.53}), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.CHOSEN and c.lr == REF
    assert c.rule == sw.RULE_WINNER
    assert c.shared_loss == 3.50 and c.tuned_loss == 3.50
    assert c.source_run == "run-lr1.2e3"


def test_the_tie_rule_keeps_the_reference_rate_for_a_gap_below_sigma():
    # The point beyond the edge exists, so the edge rule is satisfied: best is 2.4e-3, interior.
    losses = {6e-4: 3.56, 1.2e-3: 3.503, 2.4e-3: 3.500, 4.8e-3: 3.60}
    c = sw.choose_variant(pts(losses), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.CHOSEN
    assert c.best_lr == 2.4e-3
    assert c.lr == REF and c.rule == sw.RULE_TIE
    assert c.source_run == "run-lr1.2e3"
    assert c.shared_loss == 3.503 and c.tuned_loss == 3.500


def test_a_gap_above_sigma_keeps_the_tuned_rate():
    losses = {6e-4: 3.56, 1.2e-3: 3.510, 2.4e-3: 3.500, 4.8e-3: 3.60}
    c = sw.choose_variant(pts(losses), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.CHOSEN
    assert c.lr == 2.4e-3 and c.rule == sw.RULE_WINNER
    assert c.source_run == "run-lr2.4e3"
    assert c.shared_loss == 3.510 and c.tuned_loss == 3.500


def test_a_gap_equal_to_sigma_is_not_a_tie():
    losses = {6e-4: 3.75, 1.2e-3: 3.50, 2.4e-3: 3.25, 4.8e-3: 3.90}
    c = sw.choose_variant(pts(losses), reference_lr=REF, sigma_init=0.25)  # 3.50 - 3.25 is exact
    assert c.lr == 2.4e-3 and c.rule == sw.RULE_WINNER
    just_above = sw.choose_variant(pts(losses), reference_lr=REF, sigma_init=0.2500001)
    assert just_above.lr == REF and just_above.rule == sw.RULE_TIE


def test_the_edge_rule_comes_before_the_tie_rule():
    # Best is at the edge and within sigma of the reference rate: the plan's order still asks
    # for the next point before anything is chosen.
    c = sw.choose_variant(pts({6e-4: 3.56, 1.2e-3: 3.502, 2.4e-3: 3.500}), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.EXTEND
    assert c.lr is None
    assert mf.format_lr(c.next_lr) == "4.8e3"


def test_a_variant_winner_at_the_lower_edge_extends_downwards():
    c = sw.choose_variant(pts({6e-4: 3.40, 1.2e-3: 3.50, 2.4e-3: 3.53}), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.EXTEND and mf.format_lr(c.next_lr) == "3e4"


def test_a_variant_needs_its_three_points():
    c = sw.choose_variant(pts({6e-4: 3.56, 2.4e-3: 3.53}), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.INCOMPLETE and c.missing == (REF,)


def test_a_diverged_reference_point_cannot_tie():
    losses = {3e-4: 3.70, 6e-4: 3.50, 1.2e-3: None, 2.4e-3: None}
    c = sw.choose_variant(pts(losses), reference_lr=REF, sigma_init=SIGMA)
    assert c.status == sw.CHOSEN
    assert c.lr == 6e-4 and c.rule == sw.RULE_WINNER
    assert c.shared_loss is None and c.tuned_loss == 3.50


@pytest.mark.parametrize("sigma", [0.0, -0.001, float("nan"), float("inf"), None])
def test_the_tie_rule_needs_a_positive_finite_sigma(sigma):
    with pytest.raises(sw.SweepError):
        sw.choose_variant(pts({6e-4: 3.56, 1.2e-3: 3.50, 2.4e-3: 3.53}), reference_lr=REF, sigma_init=sigma)


def test_the_reference_choice_has_no_tie_rule():
    c = sw.choose_reference(pts({3e-4: 3.60, 6e-4: 3.5001, 1.2e-3: 3.50, 2.4e-3: 3.52, 4.8e-3: 3.70}))
    assert c.rule == sw.RULE_WINNER and c.shared_loss is None


# --- Reading the points from manifests ---------------------------------------------------------


def sweep_run(variant="mha", lr=1.2e-3, loss=3.5, seed=SWEEP_SEED, **kwargs):
    return fake_manifest(variant, budget="100M", branches=[], init_seed=seed, peak_lr=lr, val_loss=loss, **kwargs)


def test_points_are_the_standalone_runs_of_the_sweep_seed():
    manifests = [sweep_run(lr=lr, loss=3.5 + i * 0.01) for i, lr in enumerate(GRID)]
    manifests += [
        sweep_run(lr=1.2e-3, loss=9.0, seed=1),  # a reproducibility rerun: another seed
        sweep_run("mqa", lr=1.2e-3, loss=9.0),  # another variant
        fake_manifest("mha", init_seed=SWEEP_SEED, peak_lr=1.2e-3, val_loss=9.0),  # a 300M run
        fake_manifest("mha", init_seed=SWEEP_SEED, peak_lr=1.2e-3, val_loss=9.0, branch="100M"),  # a branch
        sweep_run(lr=1.2e-3, loss=9.0, data_seed=2),  # a data-order run
    ]
    points = sw.points_from_manifests(manifests, "mha", sweep_seed=SWEEP_SEED)
    assert [p.lr for p in points] == sorted(GRID)
    assert [p.loss for p in points] == [3.5 + i * 0.01 for i in range(5)]
    assert all(not p.diverged for p in points)
    assert [p.run for p in points] == [f"mha-isoparam-100M-lr{mf.format_lr(lr)}-s{SWEEP_SEED}-d1" for lr in sorted(GRID)]


def test_ladder_sizes_are_read_by_their_ladder_name():
    manifests = [
        fake_manifest("mha", ladder="mha_L6", budget="100M", branches=[], init_seed=SWEEP_SEED, peak_lr=lr, val_loss=3.7)
        for lr in (6e-4, 1.2e-3, 2.4e-3)
    ] + [sweep_run(lr=1.2e-3, loss=3.5)]
    assert [p.lr for p in sw.points_from_manifests(manifests, "mha_L6", sweep_seed=SWEEP_SEED)] == [6e-4, 1.2e-3, 2.4e-3]
    assert [p.lr for p in sw.points_from_manifests(manifests, "mha", sweep_seed=SWEEP_SEED)] == [1.2e-3]


def test_a_run_marked_diverged_is_a_diverged_point():
    diverged = sweep_run(lr=4.8e-3, completed=False, final_metrics={"diverged": True, "reason": "non-finite validation loss"})
    (point,) = sw.points_from_manifests([diverged], "mha", sweep_seed=SWEEP_SEED)
    assert point.diverged and point.loss is None
    assert point.reason == "non-finite validation loss"


def with_skips(manifest: dict, skipped: int) -> dict:
    """The same completed manifest with `skipped` of its steps skipped; still a valid manifest."""
    changed = copy.deepcopy(manifest)
    total = changed["steps"]["total"]
    changed["steps"] = {
        "total": total, "attempted": total, "taken": total - skipped, "skipped": skipped, "skipped_in_first_100": 0,
    }
    assert mf.validate_manifest(changed) == []
    return changed


def test_a_run_that_skipped_more_than_half_a_percent_of_its_steps_is_diverged():
    # 1,526 steps: 0.5% is 7.63, so 7 skipped steps pass and 8 do not (plan §7.3, §13.6).
    skippy = with_skips(sweep_run(lr=4.8e-3, loss=3.4), 8)
    (point,) = sw.points_from_manifests([skippy], "mha", sweep_seed=SWEEP_SEED)
    assert point.diverged and point.loss is None
    assert "skipped" in point.reason
    (fine,) = sw.points_from_manifests([with_skips(sweep_run(lr=4.8e-3, loss=3.4), 7)], "mha", sweep_seed=SWEEP_SEED)
    assert not fine.diverged and fine.loss == 3.4


def test_an_unfinished_run_is_not_a_point():
    unfinished = sweep_run(lr=2.4e-3, completed=False)
    assert sw.points_from_manifests([unfinished], "mha", sweep_seed=SWEEP_SEED) == []


def test_the_whole_path_from_manifests_to_a_choice():
    losses = {3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.52}
    manifests = [sweep_run(lr=lr, loss=loss) for lr, loss in losses.items()]
    manifests.append(sweep_run(lr=4.8e-3, completed=False, final_metrics={"diverged": True, "reason": "made up"}))
    choice = sw.choose_reference(sw.points_from_manifests(manifests, "mha", sweep_seed=SWEEP_SEED))
    assert choice.status == sw.CHOSEN and choice.lr == 1.2e-3
    assert choice.source_run == f"mha-isoparam-100M-lr1.2e3-s{SWEEP_SEED}-d1"
