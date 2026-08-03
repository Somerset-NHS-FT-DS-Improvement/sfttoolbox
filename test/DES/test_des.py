import numpy as np
import pytest

from sfttoolbox.simulation import (
    ArrivalProfile,
    CapacityPool,
    Pathway,
    PathwayStep,
    SimulationFramework,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def arrival_histogram():
    return (
        np.array([1] * 24),
        np.arange(25),
    )


@pytest.fixture
def simple_pathway():
    return [
        PathwayStep(
            "Triaged",
            5,
            "triage_room",
        ),
        PathwayStep(
            "Assessment",
            30,
            "treatment_room",
            "Triaged",
        ),
    ]


# ---------------------------------------------------------------------
# Pathway Tests
# ---------------------------------------------------------------------


def test_pathway_identifies_start_node(simple_pathway):
    pathway = Pathway(simple_pathway, "Test")

    assert pathway.start_node == "Triaged"


def test_pathway_creates_correct_nodes(simple_pathway):
    pathway = Pathway(simple_pathway, "Test")

    assert set(pathway.pathway_graph.nodes) == {
        "Triaged",
        "Assessment",
    }


def test_pathway_creates_correct_edges(simple_pathway):
    pathway = Pathway(simple_pathway, "Test")

    assert ("Triaged", "Assessment") in pathway.pathway_graph.edges


def test_find_next_step_returns_start_node(simple_pathway):
    pathway = Pathway(simple_pathway, "Test")

    node, resource, duration = pathway.find_next_step(None)

    assert node == "Triaged"
    assert resource == "triage_room"
    assert duration == 5


def test_find_next_step_returns_terminal_node(simple_pathway):
    pathway = Pathway(simple_pathway, "Test")

    current = None

    while True:
        next_node, _, _ = pathway.find_next_step(current)

        if next_node is None:
            break

        current = next_node

    assert next_node is None


# ---------------------------------------------------------------------
# Arrival Profile Tests
# ---------------------------------------------------------------------


def test_arrival_profile_calculates_lambda_max(arrival_histogram):
    profile = ArrivalProfile(
        num_patients=100,
        arrival_histogram=arrival_histogram,
    )

    assert profile.lambda_max > 0


def test_arrival_profile_acceptance_probabilities_bounded(
    arrival_histogram,
):
    profile = ArrivalProfile(
        num_patients=100,
        arrival_histogram=arrival_histogram,
    )

    assert np.all(profile.acceptance_probabilities <= 1)

    assert np.all(profile.acceptance_probabilities >= 0)


# ---------------------------------------------------------------------
# Resource Tests
# ---------------------------------------------------------------------


def test_resource_registration():
    sf = SimulationFramework()

    sf.register_resource(
        "room",
        CapacityPool(sf.env, 1),
    )

    assert "room" in sf.resources


# ---------------------------------------------------------------------
# Pathway Registration Tests
# ---------------------------------------------------------------------


def test_register_pathway(
    arrival_histogram,
    simple_pathway,
):
    sf = SimulationFramework()

    sf.register_resource(
        "triage_room",
        CapacityPool(sf.env, 1),
    )

    sf.register_resource(
        "treatment_room",
        CapacityPool(sf.env, 1),
    )

    sf.register_pathway(
        label="test",
        num_patients=100,
        arrival_rate=arrival_histogram,
        pathway=simple_pathway,
    )

    assert len(sf.pathways) == 1


def test_register_pathway_with_missing_resource_fails(
    arrival_histogram,
    simple_pathway,
):
    sf = SimulationFramework()

    sf.register_resource(
        "triage_room",
        CapacityPool(sf.env, 1),
    )

    with pytest.raises(AssertionError):
        sf.register_pathway(
            label="test",
            num_patients=100,
            arrival_rate=arrival_histogram,
            pathway=simple_pathway,
        )


# ---------------------------------------------------------------------
# Simulation Tests
# ---------------------------------------------------------------------


def test_simulation_generates_patients(
    arrival_histogram,
):
    sf = SimulationFramework()

    sf.register_resource(
        "room",
        CapacityPool(sf.env, 1),
    )

    pathway = [
        PathwayStep(
            "Assessment",
            5,
            "room",
        )
    ]

    sf.register_pathway(
        label="test",
        num_patients=50,
        arrival_rate=arrival_histogram,
        pathway=pathway,
    )

    sf.run_simulation(500)

    assert len(sf.metrics["patients"]) > 0


def test_patients_complete_pathway(
    arrival_histogram,
):
    sf = SimulationFramework()

    sf.register_resource(
        "room",
        CapacityPool(sf.env, 1),
    )

    pathway = [
        PathwayStep(
            "Assessment",
            5,
            "room",
        )
    ]

    sf.register_pathway(
        label="test",
        num_patients=50,
        arrival_rate=arrival_histogram,
        pathway=pathway,
    )

    sf.run_simulation(500)

    completed_patients = [
        patient
        for patient in sf.metrics["patients"]
        if patient.discharge_time is not None
    ]

    assert len(completed_patients) > 0


def test_resource_usage_balances(
    arrival_histogram,
):
    sf = SimulationFramework()

    sf.register_resource(
        "room",
        CapacityPool(sf.env, 1),
    )

    pathway = [
        PathwayStep(
            "Assessment",
            5,
            "room",
        )
    ]

    sf.register_pathway(
        label="test",
        num_patients=50,
        arrival_rate=arrival_histogram,
        pathway=pathway,
    )

    sf.run_simulation(500)

    events = sf.metrics["room_used"]

    net_usage = sum(delta for _, delta in events)

    assert net_usage == 0


# ---------------------------------------------------------------------
# Branching Logic Tests
# ---------------------------------------------------------------------


def test_branching_pathway():
    pathway = Pathway(
        [
            PathwayStep(
                "Start",
                1,
                "room",
            ),
            PathwayStep(
                "Route_A",
                1,
                "room",
                "Start",
                probability=0.7,
            ),
            PathwayStep(
                "Route_B",
                1,
                "room",
                "Start",
                probability=0.3,
            ),
        ],
        "test",
        seed=42,
    )

    a_count = 0
    b_count = 0

    for _ in range(10000):

        node, _, _ = pathway.find_next_step(None)

        node, _, _ = pathway.find_next_step(node)

        if node == "Route_A":
            a_count += 1
        else:
            b_count += 1

    route_a_rate = a_count / (a_count + b_count)

    assert 0.65 < route_a_rate < 0.75
