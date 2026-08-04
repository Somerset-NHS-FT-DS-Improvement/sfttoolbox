"""
Discrete Event Simulation Module.

This module provides a flexible framework for constructing and executing
discrete event simulation (DES) models using SimPy.

The framework is designed primarily for healthcare modelling but can be
applied to any system involving entities moving through a series of
resource-constrained processes.

Core Features:
    - Time-varying arrivals using non-homogeneous Poisson processes.
    - Graph-based pathway definitions using NetworkX.
    - Resource-constrained process execution using SimPy.
    - Branching pathways with probabilistic routing.
    - Fixed or stochastic activity durations.
    - Extensible resource abstractions supporting future implementations such as:
        - Time-dependent staffing.
        - Continuity of care.
        - Named clinician allocation.
        - Skill-based staffing.
        - Team-based resources.

Main Components:
    - Patient:
        Represents an individual entity progressing through the simulation.

    - PathwayStep:
        Defines a single step within a pathway, including duration,
        routing information and resource requirements.

    - Pathway:
        Directed graph representation of a care pathway.

    - ArrivalProfile:
        Defines arrival behaviour for a pathway.

    - PathwayInformation:
        Combines arrival behaviour and pathway structure.

    - ResourcePool:
        Abstract base class for resource implementations.

    - CapacityPool:
        Fixed-capacity resource implementation backed by SimPy.

    - InterarrivalCalculator:
        Generates arrivals using a non-homogeneous Poisson process.

    - SimulationFramework:
        Coordinates patient generation, pathway traversal, resource
        allocation and metric collection.

Example:
    Simulate patients attending an urgent care service consisting
    of triage, clinical assessment, diagnostic waiting and treatment.

    >>> import numpy as np
    >>> from sfttoolbox.simulation import (
    ...     PathwayStep,
    ...     SimulationFramework,
    ...     CapacityPool,
    ... )

    >>> centre = 12
    >>> scale = 3

    >>> patient_arrival_times = np.random.normal(
    ...     centre,
    ...     scale,
    ...     size=100000,
    ... )

    >>> arrival_histogram = np.histogram(
    ...     patient_arrival_times,
    ...     bins=range(24),
    ... )

    >>> patient_pathway = [
    ...     PathwayStep(
    ...         "Triaged",
    ...         5,
    ...         "triage_room",
    ...         probability=1,
    ...     ),
    ...
    ...     PathwayStep(
    ...         "Waiting for clinician to read notes",
    ...         10,
    ...         "waiting_room",
    ...         "Triaged",
    ...         1,
    ...     ),
    ...
    ...     PathwayStep(
    ...         "Assessment",
    ...         30,
    ...         "treatment_room",
    ...         "Waiting for clinician to read notes",
    ...         probability=1,
    ...     ),
    ...
    ...     PathwayStep(
    ...         "Waiting for tests",
    ...         120,
    ...         "waiting_room",
    ...         "Assessment",
    ...         1,
    ...     ),
    ...
    ...     PathwayStep(
    ...         "Treatment/Discharge",
    ...         30,
    ...         "treatment_room",
    ...         "Waiting for tests",
    ...         1,
    ...     ),
    ... ]

    >>> num_patients = 150

    >>> sf = SimulationFramework()

    >>> sf.register_resource(
    ...     "triage_room",
    ...     CapacityPool(sf.env, 3),
    ... )

    >>> sf.register_resource(
    ...     "waiting_room",
    ...     CapacityPool(sf.env, 15),
    ... )

    >>> sf.register_resource(
    ...     "treatment_room",
    ...     CapacityPool(sf.env, 10),
    ... )

    >>> sf.register_pathway(
    ...     label="Urgent Care",
    ...     num_patients=num_patients,
    ...     arrival_rate=arrival_histogram,
    ...     pathway=patient_pathway,
    ... )

    >>> sf.run_simulation(
    ...     time=24 * 60,
    ... )

Notes:
    Pathways are defined as directed graphs where nodes represent
    activities undertaken by a patient and edges represent possible
    transitions between activities.

    Resources are deliberately abstracted behind the ResourcePool
    interface, allowing future extensions to support:
        - Shift-based staffing.
        - Continuity of care.
        - Named staff assignment.
        - Skill-based routing.
        - Team allocation models.

    The framework is intended to form a reusable simulation engine
    within sfttoolbox and can be applied to outpatient, inpatient,
    emergency care, community services and wider operational modelling
    problems.
"""

__all__ = [
    "Patient",
    "PathwayStep",
    "Pathway",
    "ArrivalProfile",
    "PathwayInformation",
    "ResourcePool",
    "CapacityPool",
    "InterarrivalCalculator",
    "SimulationFramework",
]

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import simpy

logger = logging.getLogger(__name__)


@dataclass
class Patient:
    """
    Represents a patient moving through the simulation.

    Attributes:
        id (str): Unique patient identifier.
        arrival_time (Optional[float]): Simulation time at which the patient entered the system.
        discharge_time (Optional[float]): Simulation time at which the patient completed their pathway.
        pathway (List[str]): Audit trail of pathway steps completed by the patient.
    """

    id: str
    arrival_time: Optional[float] = None
    discharge_time: Optional[float] = None
    pathway: List[str] = field(default_factory=list)


@dataclass
class PathwayStep:
    """
    Represents a single step within a care pathway.

    Attributes:
        name (str): Unique step identifier.
        duration (float | Callable[[], float]): Fixed duration or callable returning a sampled duration.
        resource_label (str): Name of the resource required for this step.
        prev_step (Optional[str]): Name of the predecessor step.
        probability (float): Probability of selecting this step from its predecessor.
    """

    name: str
    duration: float | Callable[[], float]
    resource_label: str
    prev_step: Optional[str] = None
    probability: float = 1.0


@dataclass
class Pathway:
    """
    Represents a directed graph describing a patient pathway.

    Attributes:
        pathway (List[PathwayStep]): Collection of pathway steps.
        label (str): Unique identifier for the pathway.
        seed (int): Random seed used when selecting branching routes.
    """

    pathway: List[PathwayStep]
    label: str
    seed: int = 42

    def __post_init__(self) -> None:
        """
        Initialise the pathway.

        Attributes:
            pathway_graph (nx.DiGraph): Graph representation of the pathway.
            start_node (str): Name of the first node in the pathway.
            choice_rng (np.random.Generator): Random number generator used when selecting branches.
        """
        self.pathway_graph, self.start_node = self._create_pathway_graph()
        self.choice_rng = np.random.default_rng(self.seed)

    def _create_pathway_graph(self) -> tuple[nx.DiGraph, str]:
        """
        Create a directed graph representation of the pathway.

        Attributes:
            G (nx.DiGraph): Directed graph containing pathway nodes and edges.
            start_node (str): Starting node within the pathway graph.

        Returns:
            tuple[nx.DiGraph, str]: The pathway graph and the identified start node.
        """
        G = nx.DiGraph()

        nodes = []
        edges = []
        start_nodes = []
        for step in self.pathway:
            node = (
                step.name,
                {"duration": step.duration, "resource_label": step.resource_label},
            )
            edge = (step.prev_step, step.name, {"probability": step.probability})

            nodes.append(node)
            if step.prev_step is not None:
                edges.append(edge)
            else:
                start_nodes.append(node)

        assert (
            len(start_nodes) == 1
        ), f"There are multiple start points in the pathway, {start_nodes}"
        start_node = start_nodes[0][0]

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        return G, start_node

    def find_next_step(
        self,
        current_node: Optional[str],
    ) -> tuple[Optional[str], Optional[str], float]:
        """
        Determine the next step in the pathway.

        Attributes:
            current_node (Optional[str]): Current node occupied by the patient.

        Returns:
            tuple[Optional[str], Optional[str], float]:
                The next node, resource name and duration associated with the step.
        """
        if current_node is None:
            # first node
            node = self.start_node
            wait_duration = self._get_duration(node)
            resource_label = self.pathway_graph.nodes[node]["resource_label"]
        else:
            nodes_and_weights = [
                *zip(
                    *[
                        [node, node_dict["probability"]]
                        for node, node_dict in self.pathway_graph[current_node].items()
                    ]
                )
            ]

            if len(nodes_and_weights) != 0:
                nodes, weights = nodes_and_weights
                node = str(self.choice_rng.choice(nodes, p=weights))
                wait_duration = self._get_duration(node)
                resource_label = self.pathway_graph.nodes[node]["resource_label"]
            else:
                # last node
                node = None
                wait_duration = 0
                resource_label = None

        return node, resource_label, wait_duration

    def _get_duration(self, node: str) -> float:
        """
        Retrieve the duration associated with a pathway node.

        Attributes:
            node (str): Name of the pathway node.

        Returns:
            float: Duration of the node. If the duration is callable,
                the callable is evaluated and the sampled value returned.
        """
        duration = self.pathway_graph.nodes[node]["duration"]
        return duration() if callable(duration) else duration

    def plot_pathway(self, filename: str) -> None:
        """
        Generate an HTML file to visualize the graph using Mermaid.js.

        Args:
            filename (str): The name of the file where the graph visualisation will be saved.
        """
        node_numbers = {
            v: k for k, v in dict(enumerate(self.pathway_graph.nodes)).items()
        }

        graph_string = "\n".join(
            [
                self._format_edge(edge, node_numbers)
                for edge in self.pathway_graph.edges(data=True)
            ]
        )

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

        with open(filename, "w") as fout:
            fout.write(html_string)

    def _format_node(self, node_name: str, attributes: Dict[str, Any]) -> str:
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

    def _format_edge(self, edge: Any, node_numbers: Dict[Any, int]) -> str:
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

        return f"{node_numbers[src]}[{self._format_node(src, self.pathway_graph.nodes[src])}] -->{prop_string} {node_numbers[tgt]}[{self._format_node(tgt, self.pathway_graph.nodes[tgt])}]"


@dataclass
class ArrivalProfile:
    """
    Defines the arrival pattern for a pathway.

    Attributes:
        num_patients (int): Expected patient volume.
        arrival_histogram (Any): Relative arrival profile used to generate patients.
        acceptance_probabilities (Any): Acceptance probabilities for thinning.
        lambda_max (float): Maximum arrival rate used by the thinning algorithm.
    """

    num_patients: int
    arrival_histogram: Any

    def __post_init__(self) -> None:
        """
        Initialise the arrival profile.

        Attributes:
            acceptance_probabilities (Any): Acceptance probabilities used by the
                thinning algorithm for a non-homogeneous Poisson process.
            lambda_max (float): Maximum arrival rate used when generating
                candidate arrivals.
            interarrival_generator (Optional[Any]): Generator responsible for
                producing interarrival times.
        """
        self.acceptance_probabilities, self.lambda_max = (
            self._calculate_acceptance_probabilities()
        )
        self.interarrival_generator = None

    def _calculate_acceptance_probabilities(
        self,
    ) -> tuple[Any, float]:
        """
        Calculate acceptance probabilities used for arrival thinning.

        Attributes:
            mean_num_patients (Any): Expected number of patients arriving
                within each time period.
            arrival_rate (Any): Estimated arrival rate for each time period.
            lambda_max (float): Maximum arrival rate across all time periods.
            acceptance_prob (Any): Probability of accepting a candidate arrival
                in each time period.

        Returns:
            tuple[Any, float]: Acceptance probabilities and maximum arrival rate.
        """
        mean_num_patients = (
            self.arrival_histogram[0]
            / self.arrival_histogram[0].sum()
            * self.num_patients
        ).round()
        arrival_rate = mean_num_patients / 60

        lambda_max = arrival_rate.max()
        acceptance_prob = arrival_rate / lambda_max
        return acceptance_prob, lambda_max


@dataclass
class PathwayInformation:
    """
    Combines pathway and arrival information.

    Attributes:
        arrival_profile (ArrivalProfile): Arrival behaviour for the pathway.
        pathway (Pathway): Pathway definition.
        interarrival_generator (Optional[InterarrivalCalculator]):
            Generator used to sample arrivals.
    """

    arrival_profile: ArrivalProfile
    pathway: Pathway
    interarrival_generator: Optional["InterarrivalCalculator"] = field(
        default=None,
        init=False,
    )

    def set_interarrival_generator(
        self,
        interarrival_calculator: "InterarrivalCalculator",
    ) -> None:
        """
        Register an interarrival time generator.

        Attributes:
            interarrival_calculator (InterarrivalCalculator):
                Generator used to calculate interarrival times.
        """
        self.interarrival_generator = interarrival_calculator

    def calculate_interarrival_time(self) -> float:
        """
        Generate the next patient interarrival time.

        Returns:
            float: Time until the next patient arrival.
        """
        assert (
            self.interarrival_generator is not None
        ), "Please register an interarrival calculator."

        return self.interarrival_generator.calculate_interarrival_time()


class ResourcePool(ABC):
    """
    Abstract base class representing a resource pool.

    Implementations may represent:
        - Fixed-capacity resources
        - Shift-based staffing
        - Skills-based staffing
        - Continuity-of-care resources
        - Team-based resources

    All resource implementations should expose a common
    request/release interface to the simulation framework.
    """

    @abstractmethod
    def request(self, patient: Patient) -> Any:
        """
        Request a resource allocation.

        Args:
            patient (Patient): Patient requesting access.

        Returns:
            Any: Allocation token used during release.
        """
        pass

    @abstractmethod
    def release(
        self,
        allocation: Any,
    ) -> None:
        """
        Release a previously allocated resource.

        Attributes:
            allocation (Any): Allocation token returned by the request method.

        Returns:
            None: Resource is released back to the pool.
        """
        self.resource.release(allocation)


class CapacityPool(ResourcePool):
    """
    Fixed-capacity resource pool.

    Attributes:
        resource (simpy.Resource): Underlying SimPy resource.
    """

    def __init__(self, env: simpy.Environment, capacity: int) -> None:
        """
        Create a fixed-capacity resource pool.

        Args:
            env (simpy.Environment): Simulation environment.
            capacity (int): Number of concurrent users allowed.
        """
        self.resource = simpy.Resource(env, capacity)

    def request(
        self,
        patient: Patient,
    ) -> simpy.events.Event:
        """
        Request access to the resource.

        Attributes:
            patient (Patient): Patient requesting access to the resource.

        Returns:
            simpy.events.Event: SimPy request event that can be yielded
                until the resource becomes available.
        """
        return self.resource.request()

    def release(
        self,
        allocation: Any,
    ) -> None:
        """
        Release a previously allocated resource.

        Attributes:
            allocation (Any): Allocation token returned by the request
                method when the resource was acquired.

        Returns:
            None: Resource is returned to the pool and becomes available
                to other waiting patients.
        """
        self.resource.release(allocation)


class InterarrivalCalculator:
    def __init__(self, env, pathway_information, seed1=42, seed2=42):
        self.env = env

        self.lambda_max = pathway_information.arrival_profile.lambda_max
        self.total_arrival_time = len(
            pathway_information.arrival_profile.arrival_histogram[0]
        )
        self.acceptance_probabilities = (
            pathway_information.arrival_profile.acceptance_probabilities
        )

        self.exp_rng = np.random.default_rng(seed1)
        self.unif_rng = np.random.default_rng(seed2)

    def calculate_interarrival_time(self):
        interarrival_time = 0
        while True:
            interarrival_time += self.exp_rng.exponential(1 / self.lambda_max)
            candidate_time = self.env.now + interarrival_time

            clock_index = int(candidate_time // 60) % self.total_arrival_time

            u = self.unif_rng.uniform(0, 1)

            if u <= self.acceptance_probabilities[clock_index]:
                break

        return interarrival_time


class SimulationFramework:
    """
    Main simulation controller.

    Attributes:
        metrics (defaultdict[str, list]): Storage for simulation metrics.
        env (simpy.Environment): Simulation environment.
        resources (Dict[str, ResourcePool]): Registered resources.
        pathways (List[PathwayInformation]): Registered pathways.
    """

    def __init__(self) -> None:
        """
        Initialise a simulation framework.

        Attributes:
            metrics (defaultdict[str, list]): Storage for simulation metrics
                generated during execution.
            env (simpy.Environment): SimPy simulation environment.
            resources (Dict[str, ResourcePool]): Registered simulation resources.
            pathways (List[PathwayInformation]): Registered pathway definitions.
        """
        self.metrics = defaultdict(lambda: [])

        self.env = simpy.Environment()
        # setup_logger(self.env, logging.ERROR)

        self.resources = {}
        self.pathways = []

    def register_resource(
        self,
        name: str,
        resource: ResourcePool,
    ) -> None:
        """
        Register a resource with the simulation.

        Attributes:
            name (str): Unique resource identifier.
            resource (ResourcePool): Resource implementation to register.

        Returns:
            None: Resource is added to the simulation framework.
        """
        self.resources[name] = resource
        print(f"Registered new resource, {name}")

    def register_pathway(
        self,
        label: str,
        num_patients: int,
        arrival_rate: Any,
        pathway: List[PathwayStep],
        seed1: int = 42,
        seed2: int = 42,
    ) -> None:
        """
        Register a pathway with the simulation.

        Attributes:
            label (str): Unique pathway identifier.
            num_patients (int): Expected number of patients arriving on
                the pathway.
            arrival_rate (Any): Relative arrival profile used to generate
                patient arrivals.
            pathway (List[PathwayStep]): Collection of pathway steps.
            seed1 (int): Random seed used by the exponential arrival
                generator.
            seed2 (int): Random seed used by the uniform arrival
                generator.

        Returns:
            None: Pathway is added to the simulation framework.
        """
        resource_comparison = {
            pathway_step.resource_label for pathway_step in pathway
        }.difference(set(self.resources.keys()))

        assert (
            len(resource_comparison) == 0
        ), f"Resources in the pathway are not registered: {resource_comparison}"

        pathway = Pathway(pathway, label)
        arrival_profile = ArrivalProfile(num_patients, arrival_rate)

        pathway_info = PathwayInformation(arrival_profile, pathway)

        pathway_info.set_interarrival_generator(
            InterarrivalCalculator(
                self.env,
                pathway_info,
                seed1,
                seed2,
            )
        )

        self.pathways.append(pathway_info)
        print("Registered new pathway")

    def run_simulation(
        self,
        time: float,
    ) -> None:
        """
        Execute the simulation.

        Attributes:
            time (float): Simulation duration.

        Returns:
            None: Simulation is run until the specified time horizon.
        """
        # TODO: If time longer than arrivals profile,
        #       need to repeat arrival_profile!

        for pathway_info in self.pathways:
            self.env.process(self._arrival_process(pathway_info))

        self.env.run(until=time)

    def _arrival_process(
        self,
        pathway_info: PathwayInformation,
    ) -> simpy.events.Process:
        """
        Generate patients for a pathway.

        Attributes:
            pathway_info (PathwayInformation): Pathway and arrival profile
                used to generate patients.

        Returns:
            simpy.events.Process: SimPy process responsible for generating
                pathway arrivals.
        """
        patient_count = 1

        while True:
            yield self.env.timeout(pathway_info.calculate_interarrival_time())

            patient = Patient(
                f"{pathway_info.pathway.label}{patient_count}",
                self.env.now,
            )

            self.metrics["patients"].append(patient)

            logging.info(
                f"patient {pathway_info.pathway.label}"
                f"{patient_count} generated at {self.env.now / 60}"
            )

            patient_count += 1

            self.env.process(
                self._patient_process(
                    patient,
                    pathway_info,
                )
            )

    def _patient_process(
        self,
        patient: Patient,
        pathway_info: PathwayInformation,
    ) -> simpy.events.Process:
        """
        Move a patient through the pathway and allocate resources.

        Attributes:
            patient (Patient): Patient progressing through the pathway.
            pathway_info (PathwayInformation): Pathway definition and
                arrival behaviour for the patient.

        Returns:
            simpy.events.Process: SimPy process responsible for pathway
                execution and resource allocation.
        """
        current_node = None
        prev_resource = None
        prev_label = None
        prev_req = None

        while True:
            current_node, resource_label, wait_duration = (
                pathway_info.pathway.find_next_step(current_node)
            )

            if current_node is None:

                if prev_resource is not None:
                    prev_resource.release(prev_req)

                    self.metrics[f"{prev_label}_used"].append((self.env.now, -1))

                patient.discharge_time = self.env.now
                break

            resource = self.resources[resource_label]

            logging.info(f"Patient {patient.id} requesting " f"{resource_label}")

            req = resource.request(patient)
            yield req

            self.metrics[f"{resource_label}_used"].append((self.env.now, 1))

            if prev_resource is not None:
                prev_resource.release(prev_req)

                self.metrics[f"{prev_label}_used"].append((self.env.now, -1))

            logging.info(f"Patient {patient.id} waiting for " f"{wait_duration}")

            yield self.env.timeout(wait_duration)

            prev_req = req
            prev_resource = resource
            prev_label = resource_label
