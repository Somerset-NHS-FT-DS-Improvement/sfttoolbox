"""
Simulation Example 3

This example demonstrates the use of rota-based staffing resources
using the ``ShiftCapacityPool``.

Patients arrive throughout the day according to a time-varying arrival
profile and progress through a simple clinical pathway consisting of:

    Doctor triage
        ↓
    Nurse treatment
        ↓
    Doctor summary

Unlike the previous examples, resources are not available continuously.
Instead, both doctors and nurses follow a daily staffing rota with
capacity changing throughout the day.

The example illustrates:

    - Creating rota-based resources using ShiftCapacityPool.
    - Modelling workforce availability that varies by time of day.
    - Reusing the same staff resource at multiple points in a pathway.
    - Releasing staff when activities complete using
      ``release_on_completion=True``.
    - Running simulations over multiple days with cyclic staffing
      patterns.
    - Collecting utilisation and patient pathway metrics.

To run this example:

    python _example3.py

Pathway Structure:

    Doctor triage
        ↓
    Nurse treatment
        ↓
    Doctor summary

Staff Rotas:

    Doctors:

        08:00 - 10:00 : 1 doctor
        10:00 - 14:00 : 4 doctors
        14:00 - 18:00 : 2 doctors
        18:00 - 08:00 : 0 doctors

    Nurses:

        08:00 - 10:00 : 1 nurse
        10:00 - 14:00 : 4 nurses
        14:00 - 18:00 : 2 nurses
        18:00 - 08:00 : 0 nurses

Resources:

    - doctor (ShiftCapacityPool)
    - nurse (ShiftCapacityPool)

Simulation Assumptions:

    - Arrivals follow a non-homogeneous Poisson process derived
      from the supplied hourly arrival profile.
    - Activity durations are fixed.
    - Staff are released immediately after activity completion.
    - Staffing levels repeat every 24 hours.
    - Capacity reductions may be delayed if staff are currently
      engaged in activities, representing completion of work
      already in progress.

Outputs:

    After execution, simulation metrics are available through:

        sf.metrics

    This dictionary contains:

        - Patient arrival and discharge information.
        - Staff utilisation events.
        - Resource request and allocation timings.
        - Detailed patient pathway progression records.

This example demonstrates the transition from room-based modelling
towards workforce modelling and forms the basis for more advanced
representations of staffing, skills, continuity of care and rota
planning.
"""

import numpy as np

from sfttoolbox.DES import (
    PathwayStep,
    ShiftCapacityPool,
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

rota = [
    (8 * 60, 1),
    (10 * 60, 4),
    (14 * 60, 2),
    (18 * 60, 0),
]

patient_pathway = [
    PathwayStep(
        "Doctor triage",
        15,
        "doctor",
        release_on_completion=True,
    ),
    PathwayStep(
        "Nurse treatment",
        20,
        "nurse",
        "Doctor triage",
        release_on_completion=True,
    ),
    PathwayStep(
        "Doctor summary",
        20,
        "doctor",
        "Nurse treatment",
        release_on_completion=True,
    ),
]

num_patients = 40

sf = SimulationFramework()

sf.register_resource(
    "doctor",
    ShiftCapacityPool(
        sf.env,
        rota,
    ),
)

sf.register_resource(
    "nurse",
    ShiftCapacityPool(
        sf.env,
        rota,
    ),
)

sf.register_pathway(
    "standard_pathway",
    num_patients,
    arrival_histogram,
    patient_pathway,
)

# Run for two days to demonstrate the repeating rota.
sf.run_simulation(60 * 48)

# Metrics can then be accessed with:
#
#     sf.metrics
#
# Example:
#
#     sf.metrics["patients"]
#     sf.metrics["doctor_used"]
#     sf.metrics["nurse_used"]
#
# Patient-level event traces are also available via:
#
#     patient.pathway
#
# for each patient stored in:
#
#     sf.metrics["patients"]