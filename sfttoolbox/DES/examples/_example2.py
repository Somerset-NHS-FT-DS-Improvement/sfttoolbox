"""
Simulation Example 2

This example demonstrates a simulation using a directed graph where patient movement is determined by probabilities,
and nodes have capacity constraints based on available staff for appointments.

In this example, each node represents a unit where patients can have appointments. The capacity of each node is determined
by the available staff and the duration of the appointments. For instance, if a node has staff available for 8 hours in a day,
and each appointment requires 1 hour, the node can handle up to 8 appointments per day.

To run this example, use the following command in the terminal:
    python _example2.py

Make sure you have the necessary dependencies installed, including the `networkx` library.

Dependencies:
    - networkx

This example illustrates how to set up and execute a simulation that incorporates both probabilistic patient movement
and capacity constraints based on staff availability. It provides a more realistic scenario by considering resource
limitations in the healthcare system.
"""
import sfttoolbox
from dataclasses import dataclass, field
import networkx as nx
import numpy as np


# This allows a standard distribution call to take in the patient object (and does nothing with it)
@sfttoolbox.DES.distribution_wrapper
def uniform():
    return np.random.uniform()


@dataclass
class AppointmentDuration:
    time_required: int

    # This is required if using dataclasses and trying to plot the graph
    def __post_init__(self):
        self.__name__ = self.__class__.__name__


# a capacity object must include a "get" method that takes the resource and patient objects
# and an update_day method.
class StaffCapacity():
    def __init__(self):
        self.weekly_capacity = {"Mon": 2, "Tues": 2, "Weds": 2, "Thurs": 2, "Fri": 0, "Sat": 0, "Sun": 0}
        self.current_weekly_capacity = self.weekly_capacity.copy()
        self.waiting_list = []
        self.treatment_list = []

        self.wait_times = {}

        self.hours_required = None

        # Giving it a name makes the graph look nicer!
        self.__name__ = self.__class__.__name__

    def get(self, resource, patient, day_num, day):
        self.hours_required = resource.time_required
        self.treat_patient(patient, day_num, day)
        # I'm returning true here because I'll keep adding patients to the waiting list.
        return True

    def treat_patient(self, patient, day_num, day):
        if self.hours_required > self.current_weekly_capacity[day]:
            self.waiting_list.append(patient)
            self.wait_times[patient.id] = {"on": day_num}
        else:
            self.treatment_list.append(patient)
            self.current_weekly_capacity[day] -= self.hours_required

    def update_day(self, day_num, day):
        self.current_weekly_capacity = self.weekly_capacity.copy()

        patients_to_return = self.treatment_list
        self.treatment_list = []

        for idx, patient in enumerate(self.waiting_list):
            if self.hours_required > self.current_weekly_capacity[day]:
                break
            else:
                self.treatment_list.append(patient)
                self.current_weekly_capacity[day] -= self.hours_required
                self.waiting_list.pop(idx)

                self.wait_times[patient.id]["off"] = day_num

        return patients_to_return


staff_capacity = StaffCapacity()

G = nx.DiGraph()
G.add_edges_from(
    [
        ("Patient arrives", "Patient triaged"),

        ("Patient triaged", "Patient discharged", {"probability": 0.2}),
        ("Patient triaged", "Appointment made", {"probability": 0.8}),

        ("Appointment made", "Patient Treated")
    ]
)
G.add_nodes_from(
    [
        ("Patient triaged", {"distribution": uniform}),
        ("Appointment made", {"capacity": staff_capacity, "resource": AppointmentDuration(1)})
    ]
)


# id and pathway are required attributes to match the interface for the simulation
@dataclass
class Patient:
    id: int
    pathway: list[str] = field(default_factory=list)


class PatientGenerator:
    def __init__(self):
        self.id = 0

    # This is required to match the interface required for the simulation
    def generate_patients(self, day_num, day):
        patients = []
        if day == "Mon":
            for _ in range(50):
                patients.append(Patient(self.id))
                self.id += 1

        return patients


if __name__ == '__main__':
    patient_generator = PatientGenerator()

    simulation_days = 10
    sim = sfttoolbox.DES.Simulation(G, patient_generator, simulation_days)

    sim.plot_graph("sample_graph.html")

    sim.run_simulation()

    # Because we stored the wait times in the capacity object, we can now retrieve them and look at some metrics!
    number_of_patients_waiting = len(staff_capacity.wait_times)
    mean_length_of_wait = np.mean([patient["off"] - patient["on"] for patient in staff_capacity.wait_times.values() if patient.get("off")])

    print(f"{number_of_patients_waiting} patients waited an average of {mean_length_of_wait :.2f} days")