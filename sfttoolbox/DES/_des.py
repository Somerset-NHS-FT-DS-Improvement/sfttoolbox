"""
Simulation Module

This module provides a simulation framework for modeling patient flow through a directed graph representing a
healthcare system or any other system with similar dynamics.
The main class, `Simulation`, integrates with a patient generator interface to simulate the movement of patients through
the nodes of the graph, taking into account capacities and resource constraints at each node.

Classes:
    - Simulation: Manages the simulation process, including patient generation, daily updates, and graph traversal.
    - Patient: Represents a patient in the simulation, tracking their pathway through the system.
    - PatientGeneratorInterface: An interface for generating new patients at each step of the simulation.

Usage:
    1. Import necessary libraries:
       ```
       import networkx as nx
       ```
    2. Define a directed graph (`nx.DiGraph`) with nodes representing different stages or units in the system.
       Each node can have attributes like 'capacity' and 'resource'.
    3. Implement a patient generator class that adheres to the `PatientGeneratorInterface`.
    4. Initialize the `Simulation` class with the graph and the patient generator.
    5. Run the simulation using the `run_simulation` method.

For examples, see files titled "_exampleX.py" in the examples directory.

This module is designed to be flexible and extensible, allowing users to customize the graph structure, patient
attributes, and generation logic according to their specific needs.
"""
__all__ = ["Simulation", "distribution_wrapper"]

import functools
from typing import Protocol
import networkx as nx
from typing import List, Dict, Any, Optional
from itertools import cycle
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PatientInterface(Protocol):
    def __init__(self, id: int) -> None:
        """
        Initialize a patient interface with an ID and a pathway.

        Attributes:
            id (int): The patient's ID.
            pathway (List[Any]): The patient's pathway through the system.
        """
        self.id = id
        self.pathway = []


class PatientGeneratorInterface(Protocol):
    def generate_patients(self, day_num: int, day: str) -> List[PatientInterface]:
        """
        Generate a list of new patients for the given day.

        Args:
            day_num (int): The current day number in the simulation.
            day (str): The current day of the week.

        Returns:
            List[PatientInterface]: A list of newly generated patients.
        """


class CapacityInterface(Protocol):
    def get(self, resource: Any, patient: PatientInterface, day_num: int, day: str) -> bool:
        """
        Check if the capacity allows the resource allocation for the given patient.

        Args:
            resource (Any): The resource to check against capacity.
            patient (PatientInterface): The patient for whom the resource allocation is checked.
            day_num (int): The day number of the simulation.
            day (str): The day of the week.

        Returns:
            bool: True if the capacity allows resource allocation, False otherwise.
        """

    def update_day(self, day_num: int, day: str) -> List[Any]:
        """
        Update the capacity status for a given day.

        Args:
            day_num (int): The current day number in the simulation.
            day (str): The current day of the week.

        Returns:
            List[Any]: List of patients to be moved on the current day.
        """


def distribution_wrapper(func: callable) -> callable:
    """
    A decorator that wraps a distribution function to allow it fit the interface in Simulation.

    Args:
        func (callable): The distribution function to be wrapped.

    Returns:
        callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func()

    return wrapper


class Simulation:
    def __init__(self, graph: nx.DiGraph, patient_generator: PatientGeneratorInterface, number_of_days: int):
        """
        Initialize the simulation with a directed graph and a patient generator.

        Args:
            graph (nx.DiGraph): The directed graph representing the system.
            patient_generator (PatientGeneratorInterface): The patient generator interface for creating new patients.
            number_of_days (int): The number of days to run the simulation.
        """
        self.final_day_num = number_of_days
        self.graph = graph
        self.graph_checked = self.check_graph()
        self.start_node = self.identify_start_node()
        self.patient_generator = patient_generator

        self.capacities = self.collect_capacities()

        self.discharged_patients = []
        self.days_of_week = cycle(["Mon", "Tues", "Weds", "Thurs", "Fri", "Sat", "Sun"])

        self.day_num = None
        self.day = None

    def check_graph(self) -> True:
        """
        Check the graph for interfaces, probabilities, etc.

        Returns:
            bool: True if the graph passes all checks, False otherwise.
        """
        # TODO: checks interfaces, probabilities etc...
        # check if a capacity is present, a resource is also defined
        return True

    def identify_start_node(self) -> Any:
        """
        Identify the starting node in the graph.

        Returns:
            Any: The starting node.

        Raises:
            AttributeError: If no starting node is found.
        """
        for node, degree in self.graph.in_degree:
            if degree == 0:
                starting_node = node
                break
        else:
            # TODO: in a scenario where something is fed back to the start this won't work...
            raise AttributeError("No starting node")

        return starting_node

    def collect_capacities(self) -> Dict[Any, Any]:
        """
        Collect the capacities from the graph nodes.

        Returns:
            Dict[Any, Any]: A dictionary of capacities for each node.
        """
        return nx.get_node_attributes(self.graph, "capacity")

    def run_simulation(self) -> None:
        """
        Run the simulation for the specified number of days.
        """
        # TODO: update day, go through all the patients on the books and run the relevant update days
        # TODO: generate patients
        # TODO: run each patient down the graph until assigned a capacity or discharged

        for day_num, day in zip(range(0, self.final_day_num, 1), self.days_of_week):
            self.day_num = day_num
            self.day = day
            logger.info(f"day number: {day_num}, day: {day}")

            patients_to_move = {node: capacity.update_day(day_num, day) for node, capacity in self.capacities.items()}
            if any(patients_to_move.values()):
                # TODO: use itertools here
                for node, patients in patients_to_move.items():
                    for patient in patients:
                        logger.info(f"Moving previous patient: patient {patient.id}")

                        discharged_patient = self.traverse_graph(node, patient, check_capacity=False)
                        # TODO: abstract this out
                        if discharged_patient:
                            self.discharged_patients.append(discharged_patient)

            new_patients = self.patient_generator.generate_patients(day_num, day)

            node = self.start_node
            for patient in new_patients:
                logger.info(f"New patient generated: patient {patient.id}")

                discharged_patient = self.traverse_graph(node, patient)
                if discharged_patient:
                    self.discharged_patients.append(discharged_patient)

    def traverse_graph(self, node: Any, patient: PatientInterface, check_capacity: bool = True) -> Optional[
        PatientInterface]:
        """
        Traverse the graph for a given patient starting from a specified node.

        Args:
            node (Any): The current node in the graph.
            patient (Patient): The patient being processed.
            check_capacity (bool, optional): Whether to check capacity constraints. Defaults to True.

        Returns:
            Optional[Patient]: The patient if they are discharged, otherwise None.
        """
        node_attrs = self.graph.nodes[node]
        next_nodes, edge_attrs = [*zip(*self.graph[node].items())]

        capacity = node_attrs.get("capacity")
        if check_capacity and capacity:
            resource = node_attrs["resource"]

            if not capacity.get(resource=resource, patient=patient, day_num=self.day_num, day=self.day):

                for edge_index, edge_attr in enumerate(edge_attrs):
                    if "capacity" in edge_attr:
                        break
                next_node = next_nodes[edge_index]
            else:
                return

        else:

            patient.pathway.append(node)

            if len(next_nodes) == 1:
                next_node = next_nodes[0]
            else:

                prob = node_attrs.get("distribution")(patient)

                traverse_probs = np.cumsum([edge_attr.get("probability", 0) for edge_attr in edge_attrs])

                if traverse_probs[-1] != 1:

                    bernoulli = [edge_attr.get("bernoulli") for edge_attr in edge_attrs]

                    if any(bernoulli):
                        assert len(
                            bernoulli) == 2, f"When using Bernoulli, there should only be 2 options, check node {node}"
                        next_node = next_nodes[0] if bernoulli[0] == prob else next_nodes[1]

                    else:
                        raise ValueError(
                            f"Probabilities of pathway must add up to 1 or contain a Bernoulli trial, check node {node}")

                else:
                    next_node = next_nodes[np.searchsorted(traverse_probs, prob)]

        logger.info(f"Moving to next node: {next_node}")

        if self.graph.out_degree[next_node] > 0 and next_node not in self.capacities.keys():
            return self.traverse_graph(next_node, patient)
        else:
            # Final node reached
            patient.pathway.append(next_node)
            return patient

    def plot_graph(self, filename: str) -> None:
        """
        Generate an HTML file to visualize the graph using Mermaid.js.

        Args:
            filename (str): The name of the file where the graph visualization will be saved.
        """
        node_numbers = {v: k for k, v in dict(enumerate(self.graph.nodes)).items()}

        graph_string = "\n".join([self.__format_edge(edge, node_numbers) for edge in self.graph.edges(data=True)])

        html_string = f"""
        <html>
        <body>
        
        <style>
            .node rect {{
                fill: #edae49 !important;
                stroke: #edae49 !important;
            }}
        </style>
        <pre class="mermaid">
                    graph TD
                    {graph_string}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        </body>
        </html>
        """

        with open(filename, 'w') as fout:
            fout.write(html_string)

    def __format_node(self, node_name: str, attributes: Dict[str, Any]) -> str:
        """
        Format the node information for graph visualization.

        Args:
            node_name (str): The name of the node.
            attributes (Dict[str, Any]): The attributes of the node.

        Returns:
            str: The formatted string representation of the node.
        """
        atts = []
        for k, v in attributes.items():
            v = v.__name__ if hasattr(v, "__name__") else v
            atts.append(f"{k}: {v}")
        atts = "\n".join(atts)
        return f"{node_name}\n{atts}"

    def __format_edge(self, edge: Any, node_numbers: Dict[Any, int]) -> str:
        """
        Format the edge information for graph visualization.

        Args:
            edge (Any): The edge in the graph, including source, target, and properties.
            node_numbers (Dict[Any, int]): A mapping of node names to their corresponding numbers.

        Returns:
            str: The formatted string representation of the edge.
        """
        src, tgt, props = edge

        prop_string = ""
        if props:
            prop_string = "|" + "\n".join([f"{k}: {v}" for k, v in props.items()]) + "|"

        return f"{node_numbers[src]}[{self.__format_node(src, self.graph.nodes[src])}] -->{prop_string} {node_numbers[tgt]}[{self.__format_node(tgt, self.graph.nodes[tgt])}]"
