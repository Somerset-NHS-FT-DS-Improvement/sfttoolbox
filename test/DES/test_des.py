import pytest
import networkx as nx
from unittest.mock import Mock
from pathlib import Path
from typing import Any, List

import sfttoolbox


# Mock classes
class MockPatientGenerator:
    def generate_patients(self, day_num: int, day: str) -> List['MockPatient']:
        return []

class MockPatient:
    def __init__(self, patient_id: int):
        self.id = patient_id
        self.pathway = []

class MockCapacity:
    def get(self, resource: Any, patient: MockPatient, day_num: int, day: str) -> bool:
        return True

    def update_day(self, day_num: int, day: str) -> List:
        return []

@pytest.fixture
def graph() -> nx.DiGraph:
    """
    Fixture to create a mock directed graph for testing.

    Returns:
        nx.DiGraph: The mock directed graph.
    """
    G = nx.DiGraph()
    G.add_node("Start", capacity=MockCapacity(), resource="bed", distribution=Mock(return_value=0.5))
    G.add_node("Middle", capacity=MockCapacity(), resource="bed", distribution=Mock(return_value=0.5))
    G.add_node("End")
    G.add_edge("Start", "Middle", probability=1)
    G.add_edge("Middle", "End", probability=1)
    return G

@pytest.fixture
def patient_generator() -> MockPatientGenerator:
    """
    Fixture to create a mock patient generator for testing.

    Returns:
        MockPatientGenerator: The mock patient generator.
    """
    return MockPatientGenerator()

@pytest.fixture
def simulation(graph: nx.DiGraph, patient_generator: MockPatientGenerator) -> sfttoolbox.DES.Simulation:
    """
    Fixture to create a Simulation object for testing.

    Args:
        graph (nx.DiGraph): The directed graph.
        patient_generator (MockPatientGenerator): The mock patient generator.

    Returns:
        Simulation: The Simulation object.
    """
    return sfttoolbox.DES.Simulation(graph, patient_generator, 7)

def test_check_graph(simulation: sfttoolbox.DES.Simulation) -> None:
    """
    Test to ensure the check_graph method returns True.

    Args:
        simulation (Simulation): The Simulation object.
    """
    assert simulation.check_graph() is True

def test_identify_start_node(simulation: sfttoolbox.DES.Simulation) -> None:
    """
    Test to ensure the start node is correctly identified.

    Args:
        simulation (Simulation): The Simulation object.
    """
    assert simulation.identify_start_node() == "Start"

def test_collect_capacities(simulation: sfttoolbox.DES.Simulation) -> None:
    """
    Test to ensure capacities are correctly collected from the graph.

    Args:
        simulation (Simulation): The Simulation object.
    """
    capacities = simulation.collect_capacities()
    assert "Start" in capacities
    assert "Middle" in capacities

def test_run_simulation(simulation: sfttoolbox.DES.Simulation) -> None:
    """
    Test to ensure the simulation runs correctly and updates day_num and day.

    Args:
        simulation (Simulation): The Simulation object.
    """
    simulation.run_simulation()
    assert simulation.day_num == 6  # Should be the last day (0-6 for 7 days)
    assert simulation.day == "Sun"

def test_traverse_graph(simulation: sfttoolbox.DES.Simulation) -> None:
    """
    Test to ensure a patient traverses the graph correctly.

    Args:
        simulation (Simulation): The Simulation object.
    """
    patient = MockPatient(patient_id=1)
    result = simulation.traverse_graph("Start", patient, check_capacity=False)
    assert patient.pathway == ["Start"]

def test_plot_graph(simulation: sfttoolbox.DES.Simulation, tmp_path: Path) -> None:
    """
    Test to ensure the graph visualization is created and saved.

    Args:
        simulation (Simulation): The Simulation object.
        tmp_path (Path): The temporary path for saving the graph visualization.
    """
    filename = tmp_path / "graph.html"
    simulation.plot_graph(filename)
    assert filename.exists()