"""
Simulation Example 1

This example demonstrates the basic use of the simulation framework
to model patients moving through a simple healthcare pathway.

Patients arrive throughout the day according to a time-varying
arrival profile and progress through three stages:

    1. Triage
    2. Assessment
    3. Treatment

Each stage requires a resource with finite capacity. If all
resources are occupied, patients queue until capacity becomes
available.

The example illustrates:

    - Creating a patient pathway.
    - Defining an arrival profile.
    - Registering simulation resources.
    - Registering a pathway.
    - Running a simulation.
    - Accessing simulation metrics.

To run this example:

    python _example1.py

Simulation Structure:

    Triaged
        ↓
    Assessment
        ↓
    Treatment

Resources:

    - triage_room (capacity = 3)
    - assessment_room (capacity = 15)
    - treatment_room (capacity = 10)

Simulation Assumptions:

    - Arrivals follow a non-homogeneous Poisson process derived
      from the supplied hourly arrival profile.
    - Activity durations are fixed.
    - Patients follow a single deterministic pathway.
    - Resource capacities constrain patient flow.

Outputs:

    After execution, simulation metrics are available through:

        sf.metrics

    This dictionary contains:

        - Patient information.
        - Resource utilisation events.
        - Pathway completion information.

This example is intended as a minimal working example before
adding more advanced features such as branching pathways,
stochastic activity durations, multiple pathways and
alternative resource allocation strategies.
"""

import numpy as np

from sfttoolbox.DES import (
    CapacityPool,
    PathwayStep,
    SimulationFramework,
)

centre = 12
scale = 3

patient_arrival_times = np.random.normal(
    centre,
    scale,
    size=100000,
)

arrival_histogram = np.histogram(
    patient_arrival_times,
    bins=range(24),
)

patient_pathway = [
    PathwayStep(
        "Triaged",
        15,
        "triage_room",
    ),
    PathwayStep(
        "Assessment",
        20,
        "assessment_room",
        "Triaged",
    ),
    PathwayStep("Treatment", 45, "treatment_room", "Assessment"),
]

num_patients = 150

sf = SimulationFramework()

sf.register_resource("triage_room", CapacityPool(sf.env, 3))
sf.register_resource("assessment_room", CapacityPool(sf.env, 15))
sf.register_resource("treatment_room", CapacityPool(sf.env, 10))

sf.register_pathway(
    "standard_pathway", num_patients, arrival_histogram, patient_pathway
)

sf.run_simulation(60 * 24)

# metrics can then be accessed with sf.metrics
