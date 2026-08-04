"""
Simulation Example 2

This example demonstrates a pathway containing a probabilistic branch.

Patients arrive throughout the day according to a time-varying arrival
profile and progress through triage and assessment before following
one of two possible routes:

    Route 1 (50%):
        Assessment
            ↓
        Treatment

    Route 2 (50%):
        Assessment
            ↓
        Waiting for tests
            ↓
        Treatment post tests

The route taken by each patient is selected using the probabilities
defined on the outgoing pathway transitions.

The example illustrates:

    - Creating branching patient pathways.
    - Defining probabilistic routing between pathway steps.
    - Creating a time-varying arrival profile.
    - Registering multiple simulation resources.
    - Running a pathway with alternative patient journeys.
    - Collecting pathway and utilisation metrics.

To run this example:

    python _example2.py

Pathway Structure:

    Triaged
        ↓
    Assessment
       / \
      /   \
     ↓     ↓
Treatment  Waiting for tests
               ↓
       Treatment post tests

Resources:

    - triage_room (capacity = 3)
    - assessment_room (capacity = 15)
    - treatment_room (capacity = 10)
    - waiting_room (capacity = 25)

Simulation Assumptions:

    - Arrivals follow a non-homogeneous Poisson process derived
      from the supplied hourly arrival profile.
    - Activity durations are fixed.
    - Patients follow one of two treatment routes.
    - Resource capacities constrain patient flow.
    - Routing probabilities from Assessment sum to 1.

Outputs:

    After execution, simulation metrics are available through:

        sf.metrics

    This dictionary contains:

        - Patient arrival and discharge information.
        - Resource utilisation events.
        - Pathway progression information.

This example introduces probabilistic branching while retaining a
simple pathway structure. It serves as a foundation for modelling
more complex healthcare pathways involving alternative treatment
routes, diagnostics, referrals, rework loops and discharge pathways.
"""

import numpy as np

from sfttoolbox.simulation import (
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
    PathwayStep("Treatment", 45, "treatment_room", "Assessment", probability=0.5),
    PathwayStep("Waiting for tests", 60, "waiting_room", "Assessment", probability=0.5),
    PathwayStep(
        "Treatment post tests",
        20,
        "treatment_room",
        "Waiting for tests",
    ),
]

num_patients = 150

sf = SimulationFramework()

sf.register_resource("triage_room", CapacityPool(sf.env, 3))
sf.register_resource("assessment_room", CapacityPool(sf.env, 15))
sf.register_resource("treatment_room", CapacityPool(sf.env, 10))
sf.register_resource("waiting_room", CapacityPool(sf.env, 25))

sf.register_pathway(
    "standard_pathway", num_patients, arrival_histogram, patient_pathway
)

sf.run_simulation(60 * 24)

# metrics can then be accessed with sf.metrics
