"""
Tests for PhaseTracker — pure Python, no GPU needed.

Three cases per spec:
  (a) Normal 0→1→2 transition with correct step detection
  (b) patience not met → no transition
  (c) Flapping near threshold → monotone (never regresses)
"""

import sys
import os

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.z_logging import PhaseTracker


THETA = 0.999
PATIENCE = 2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_sequence(tracker: PhaseTracker, sequence):
    """
    Feed a list of (train_exact, test_exact, step) tuples to the tracker.
    Returns list of (phase, transitioned) after each update.
    """
    results = []
    for train_acc, test_acc, step in sequence:
        phase, transitioned = tracker.update(train_acc, test_acc, step=step)
        results.append((phase, transitioned))
    return results


# ---------------------------------------------------------------------------
# Case (a): Normal 0→1→2 transition, step detection
# ---------------------------------------------------------------------------

def test_normal_0_1_2_transition():
    """
    Sequence:
      steps 1-3:  train < theta  → phase 0
      steps 4-6:  train ≥ theta, test < theta (patience=2 met at step 5) → phase 1
      steps 7-9:  both ≥ theta (patience=2 met at step 8) → phase 2
    """
    tracker = PhaseTracker(phase_threshold=THETA, phase_patience=PATIENCE)

    seq = [
        # Phase 0 region
        (0.5, 0.4, 1),
        (0.7, 0.6, 2),
        (0.9, 0.8, 3),
        # Phase 1 candidate starts at step 4
        (1.0, 0.5, 4),
        (1.0, 0.5, 5),  # patience fulfilled → transition to phase 1
        (1.0, 0.5, 6),
        # Phase 2 candidate starts at step 7
        (1.0, 1.0, 7),
        (1.0, 1.0, 8),  # patience fulfilled → transition to phase 2
        (1.0, 1.0, 9),
    ]

    results = run_sequence(tracker, seq)

    # Steps 1-3: still phase 0
    assert all(p == 0 for p, _ in results[:3]), f"Expected phase 0 for steps 1-3, got {results[:3]}"

    # Step 4: candidate started, patience not met yet
    assert results[3] == (0, False), f"At step 4 (1st candidate eval) expected (0, False), got {results[3]}"

    # Step 5: patience met → transition to phase 1
    assert results[4] == (1, True), f"At step 5 expected transition to phase 1, got {results[4]}"

    # Step 6: still phase 1, no new transition
    assert results[5] == (1, False), f"At step 6 expected (1, False), got {results[5]}"

    # Step 7: phase 2 candidate, patience not met
    assert results[6] == (1, False), f"At step 7 expected (1, False), got {results[6]}"

    # Step 8: patience met → transition to phase 2
    assert results[7] == (2, True), f"At step 8 expected transition to phase 2, got {results[7]}"

    # Step 9: still phase 2
    assert results[8] == (2, False), f"At step 9 expected (2, False), got {results[8]}"

    # Verify transition_steps history
    assert tracker.transition_steps == [5, 8], f"Expected transition steps [5,8], got {tracker.transition_steps}"


# ---------------------------------------------------------------------------
# Case (b): patience not met → no transition
# ---------------------------------------------------------------------------

def test_patience_not_met_no_transition():
    """
    With patience=3, only 2 consecutive evals above threshold → no transition.
    """
    tracker = PhaseTracker(phase_threshold=THETA, phase_patience=3)

    seq = [
        (0.5, 0.4, 1),   # phase 0
        (1.0, 0.5, 2),   # candidate phase 1, count=1
        (1.0, 0.5, 3),   # count=2 — still < patience=3
        (0.5, 0.4, 4),   # drops back → resets candidate
    ]

    results = run_sequence(tracker, seq)

    assert all(p == 0 for p, _ in results), f"Expected all phase 0, got {results}"
    assert all(not t for _, t in results), "Expected no transitions"
    assert tracker.transition_steps == []


# ---------------------------------------------------------------------------
# Case (c): flapping near threshold → monotone (no regression)
# ---------------------------------------------------------------------------

def test_flapping_near_threshold_monotone():
    """
    Once a transition to phase 1 is committed, flapping below threshold
    must not regress the phase back to 0.

    Also tests that a subsequent transition to phase 2 works normally.
    """
    tracker = PhaseTracker(phase_threshold=THETA, phase_patience=PATIENCE)

    # Phase 1 transition
    seq_phase1 = [
        (1.0, 0.5, 1),
        (1.0, 0.5, 2),  # → phase 1 committed
    ]
    results = run_sequence(tracker, seq_phase1)
    assert results[-1] == (1, True), f"Expected phase 1 transition at step 2, got {results[-1]}"

    # Flap back to below threshold
    seq_flap = [
        (0.5, 0.4, 3),  # raw would be phase 0, but committed is 1
        (0.5, 0.4, 4),
        (0.5, 0.4, 5),
    ]
    results_flap = run_sequence(tracker, seq_flap)
    assert all(p == 1 for p, _ in results_flap), \
        f"Phase should not regress below committed level 1, got {results_flap}"
    assert all(not t for _, t in results_flap), "No new transitions expected during flapping"

    # Phase 2 transition from recovered values
    seq_phase2 = [
        (1.0, 1.0, 6),
        (1.0, 1.0, 7),  # → phase 2 committed
    ]
    results_p2 = run_sequence(tracker, seq_phase2)
    assert results_p2[-1] == (2, True), f"Expected phase 2 transition at step 7, got {results_p2[-1]}"

    # Verify final committed phase
    assert tracker.phase == 2
    assert tracker.transition_steps == [2, 7]
