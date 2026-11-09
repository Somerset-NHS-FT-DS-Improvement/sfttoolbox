""" Simulation Example 4: Rota-Based Staffing with Multiple Replications

This example demonstrates rota-based staffing resources using ShiftCapacityPool and shows how resource utilisation can be analysed across multiple simulation replications.

Patients arrive according to a time-varying arrival profile and progress through a simple clinical pathway:

Doctor triage
    ↓
Nurse treatment
    ↓
Doctor summary

Unlike resources with constant capacity, the doctors and nurses in this example follow a daily staffing rota. Resource capacity therefore changes throughout the simulated day and the rota repeats every 24 hours.

The example illustrates:

- Creating rota-based resources using ``ShiftCapacityPool``.
- Modelling workforce availability that varies by time of day.
- Reusing the same staff resource at multiple points in a pathway.
- Releasing staff after activities using
  ``release_on_completion=True``.
- Running simulations over multiple days with cyclic staffing patterns.
- Running multiple independent simulation replications.
- Using reproducible random seeds for each replication.
- Combining resource utilisation results across replications.
- Visualising median resource utilisation and its uncertainty across
  replications using ``SimulationAnalyses``.

Pathway Structure:

Doctor triage
    ↓
Nurse treatment
    ↓
Doctor summary

Staff Rotas:

Doctors and nurses:

    08:00 - 10:00 : capacity 1
    10:00 - 14:00 : capacity 4
    14:00 - 18:00 : capacity 2
    18:00 - 08:00 : capacity 0

Resources:

- doctor (ShiftCapacityPool)
- nurse (ShiftCapacityPool)

Simulation Configuration:

- Simulation duration: 48 hours.
- Number of patients per replication: 40.
- Number of simulation replications: 100.
- Each replication receives two reproducible random seeds generated
  from a NumPy ``SeedSequence`` initialised with seed 42.
- The same staffing rota is used for doctors and nurses.
- Staffing levels repeat every 24 hours.

Simulation Assumptions:

- Arrivals follow a non-homogeneous Poisson process derived from the
  supplied hourly arrival profile.
- The arrival profile is generated from a normal distribution centred
  around midday.
- Activity durations are fixed.
- Staff are released immediately after activity completion.
- Capacity reductions may be delayed while staff are engaged in
  activities, allowing work already in progress to complete.

Analysis:

Each completed ``SimulationFramework`` instance is stored and passed
to ``SimulationAnalyses``.

Resource utilisation data are generated separately for doctors and
nurses using ``create_resource_dataframe``. The replications are aligned
onto a common simulation-time index before being plotted.

``plot_resource_usage`` summarises utilisation across the simulation
replications using the median and a 95% percentile interval.

Outputs:

Metrics for each individual simulation remain available through:

    sf.metrics

The collection of completed simulation replications is stored in:

    sims

The final plot compares doctor and nurse utilisation across all
replications.

To run this example:

python _example4.py

This example extends rota-based workforce modelling to stochastic experimentation across multiple simulation replications. It provides a basis for exploring staffing requirements, resource utilisation, uncertainty, skills, continuity of care and alternative rota designs. 
"""



import numpy as np

from sfttoolbox.DES import (
    PathwayStep,
    ShiftCapacityPool,
    SimulationAnalyses,
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

sim_runtime = 60 * 48

num_patients = 40

n_runs = 100
seeds = np.random.SeedSequence(42).generate_state(n_runs * 2).reshape(-1, 2)
sims = []

for seed_collection in seeds:
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
        seed1=seed_collection[0],
        seed2=seed_collection[1],
    )

    sf.run_simulation(sim_runtime)

    sims.append(sf)

sa = SimulationAnalyses(sims)
dfs = [
    sa.create_resource_dataframe("doctor_used"),
    sa.create_resource_dataframe("nurse_used"),
]

_ = sa.plot_resource_usage(dfs, labels=["doctor", "nurse"])
