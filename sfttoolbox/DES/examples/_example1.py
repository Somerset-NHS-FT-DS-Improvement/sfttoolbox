"""
Simulation Example 1

This example demonstrates a simple simulation using a directed graph where patient movement is determined by
probabilities. This example does not take node capacity into account.

To run this example, use the following command in the terminal:
    python _example1.py

Make sure you have the necessary dependencies installed, including the `networkx` library.

Dependencies:
    - networkx

This example helps to illustrate the basic setup and execution of a simulation without capacity constraints, focusing
on probabilistic patient movement through the system.
"""
import sfttoolbox
from dataclasses import dataclass, field
import networkx as nx
import numpy as np


# This allows a standard distribution call to take in the patient object (and does nothing with it)
@sfttoolbox.DES.distribution_wrapper
def uniform():
    return np.random.uniform()


# Create a simple graph
G = nx.DiGraph()
G.add_edges_from(
    [
        ("Patient arrives", "Patient triaged"),

        # Probabilities determine the chance of moving
        ("Patient triaged", "Patient discharged", {"probability": 0.2}),
        ("Patient triaged", "Appointment made", {"probability": 0.8}),

        ("Appointment made", "Patient Treated")
    ]
)
G.add_nodes_from(
    [
        # Defining a distribution to generate probabilities.
        ("Patient triaged", {"distribution": uniform})
    ]
)

# id and pathway are required attributes to match the interface for the simulation
@dataclass
class Patient:
    id: int
    pathway: list[str] = field(default_factory=list)

# generate_patients is required to match the interface required for the simulation
class PatientGenerator:
    def __init__(self):
        self.id = 0


    def generate_patients(self, day_num, day):
        patients = []
        if day == "Mon":
            for _ in range(5):
                patients.append(Patient(self.id))
                self.id += 1

        return patients


if __name__ == '__main__':
    patient_generator = PatientGenerator()

    number_of_simulation_days = 10
    sim = sfttoolbox.DES.Simulation(G, patient_generator, number_of_simulation_days)

    # Create an html file of the graph.
    sim.plot_graph("sample_graph.html")
    sim.run_simulation()

    # We can then create a patient flow out of the discharged patients
    G2 = nx.DiGraph()

    for patient in sim.discharged_patients:

        edges = [(patient.pathway[i], patient.pathway[i+1]) for i in range(len(patient.pathway) -1)]

        # Create a graph out of the discharged patient pathway
        G2.add_edges_from(edges)

        # Add colour and value attributes to the edges
        for edge in edges:
            G2.edges[edge]["value"] = G2.edges[edge].get("value", 0) + 1
            G2.edges[edge]["color"] = "blue"

    # Use our convenient sankey generator to view the flow
    sfttoolbox.plotting.generate_sankey(G2)
#%%
