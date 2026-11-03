import pytest

from optimiser import Objective, build_circuit, num_settings, train

TRIANGLE_EDGES = [(0, 1), (1, 2), (2, 0)]
SINGLE_EDGE = [(0, 1)]
LINE_EDGES = [(0, 1), (1, 2)]
SEEDS = [0, 1, 2, 3, 4]


def test_build_circuit_rejects_too_many_settings():
    with pytest.raises(ValueError):
        build_circuit("xqaoa", [0.0] * 10, 3, TRIANGLE_EDGES)


def test_build_circuit_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_circuit("vqe", [0.0, 0.0], 3, TRIANGLE_EDGES)  # pyright: ignore[reportArgumentType]


def test_number_of_settings():
    assert num_settings("qaoa", 3, LINE_EDGES) == 2
    assert num_settings("xqaoa", 3, LINE_EDGES) == len(LINE_EDGES) + 2 * 3


def test_objective_counts_evaluations_and_return_negative():
    objective = Objective("qaoa", 2, SINGLE_EDGE)

    first = objective([0.0, 0.0])
    second = objective([0.1, 0.2])

    assert first == pytest.approx(-0.5)
    assert objective.n_evaluations == 2
    assert objective.history[0] == pytest.approx(0.5)
    assert second == pytest.approx(-objective.history[1])


@pytest.mark.parametrize("seed", SEEDS)
def test_qaoa_trains_to_perfect_score_on_single_edge(seed):
    result = train("qaoa", 2, SINGLE_EDGE, max_evaluations=200, seed=seed)

    assert result["expected_score"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_xqaoa_trains_to_perfect_score_on_line(seed):
    result = train("xqaoa", 3, LINE_EDGES, max_evaluations=500, seed=seed)

    assert result["expected_score"] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_qaoa_stays_below_best_on_line(seed):
    """One-layer QAOA tops out near 1.65 (82%) on the line, below the best of 2."""
    result = train("qaoa", 3, LINE_EDGES, max_evaluations=200, seed=seed)

    assert 1.5 < result["expected_score"] < 1.66


@pytest.mark.parametrize("method", ["qaoa", "xqaoa"])
def test_training_improves_on_start(method):
    result = train(method, 3, LINE_EDGES, max_evaluations=200, seed=0)

    assert result["expected_score"] > result["history"][0]


def test_training_never_exceeds_evaluation_limit():
    result = train("xqaoa", 3, LINE_EDGES, max_evaluations=50, seed=0)

    assert result["n_evaluations"] <= 50
    assert len(result["history"]) == result["n_evaluations"]


def test_same_seed_gives_same_result():
    a = train("xqaoa", 3, LINE_EDGES, max_evaluations=100, seed=7)
    b = train("xqaoa", 3, LINE_EDGES, max_evaluations=100, seed=7)

    assert a["initial_settings"] == b["initial_settings"]
    assert a["final_settings"] == b["final_settings"]
