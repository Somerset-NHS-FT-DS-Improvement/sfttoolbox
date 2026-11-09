from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

matplotlib.use("Agg")

from sfttoolbox.simulation import SimulationAnalyses


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close figures after every test to prevent cross-test contamination."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def mock_team_colours(monkeypatch):
    """Provide deterministic colours without depending on toolbox configuration."""
    monkeypatch.setattr(
        sfttoolbox,
        "team_colours",
        {
            "blue": {"hex": "#0072B2"},
            "orange": {"hex": "#E69F00"},
            "green": {"hex": "#009E73"},
        },
    )


def make_simulation(now, **metrics):
    """Create a minimal simulation-framework test double."""
    return SimpleNamespace(
        env=SimpleNamespace(now=now),
        metrics=metrics,
    )


class TestInitialisation:
    def test_initialises_from_single_simulation(self):
        simulation = make_simulation(now=121)

        analysis = SimulationAnalyses(simulation)

        assert analysis.sf is simulation
        assert analysis.final_time_mins == 121
        assert analysis.final_time == 3

    def test_initialises_from_multiple_simulations_using_longest_runtime(self):
        simulations = [
            make_simulation(now=60),
            make_simulation(now=121),
            make_simulation(now=90),
        ]

        analysis = SimulationAnalyses(simulations)

        assert analysis.sf is simulations
        assert analysis.final_time_mins == 121
        assert analysis.final_time == 3

    def test_rounds_partial_hour_up(self):
        simulation = make_simulation(now=61)

        analysis = SimulationAnalyses(simulation)

        assert analysis.final_time == 2

    def test_exact_hour_is_not_rounded_further(self):
        simulation = make_simulation(now=120)

        analysis = SimulationAnalyses(simulation)

        assert analysis.final_time == 2


class TestCreateResourceDataframeSingleSimulation:
    def test_creates_cumulative_resource_usage(self):
        simulation = make_simulation(
            now=30,
            doctor_used=[
                (10, 1),
                (20, 2),
                (30, -1),
            ],
        )
        analysis = SimulationAnalyses(simulation)

        result = analysis.create_resource_dataframe("doctor_used")

        expected = pd.DataFrame(
            {
                "Time": [0, 10, 20, 30],
                "doctor_used": [0, 1, 2, -1],
                "cumsum": [0, 1, 3, 2],
            }
        )

        assert_frame_equal(result.reset_index(drop=True), expected)

    def test_sums_resource_changes_at_the_same_time(self):
        simulation = make_simulation(
            now=30,
            doctor_used=[
                (10, 1),
                (10, 2),
                (20, -1),
                (20, 3),
            ],
        )
        analysis = SimulationAnalyses(simulation)

        result = analysis.create_resource_dataframe("doctor_used")

        expected = pd.DataFrame(
            {
                "Time": [0, 10, 20],
                "doctor_used": [0, 3, 2],
                "cumsum": [0, 3, 5],
            }
        )

        assert_frame_equal(result.reset_index(drop=True), expected)

    def test_supports_usage_falling_back_to_zero(self):
        simulation = make_simulation(
            now=20,
            nurse_used=[
                (5, 2),
                (10, 1),
                (20, -3),
            ],
        )
        analysis = SimulationAnalyses(simulation)

        result = analysis.create_resource_dataframe("nurse_used")

        assert result["cumsum"].tolist() == [0, 2, 3, 0]

    def test_raises_key_error_for_unknown_metric(self):
        simulation = make_simulation(
            now=30,
            doctor_used=[(10, 1)],
        )
        analysis = SimulationAnalyses(simulation)

        with pytest.raises(KeyError, match="nurse_used"):
            analysis.create_resource_dataframe("nurse_used")

    def test_empty_metrics_returns_initial_zero_row(self):
        simulation = make_simulation(
            now=30,
            doctor_used=[],
        )
        analysis = SimulationAnalyses(simulation)

        result = analysis.create_resource_dataframe("doctor_used")

        expected = pd.DataFrame(
            {
                "Time": [0],
                "doctor_used": [0],
                "cumsum": [0],
            }
        )

        assert_frame_equal(
            result.reset_index(drop=True),
            expected,
            check_dtype=False,
        )


class TestCreateResourceDataframeMultipleSimulations:
    def test_combines_replications_on_a_common_minute_index(self):
        simulations = [
            make_simulation(
                now=5,
                doctor_used=[
                    (1, 1),
                    (3, 1),
                ],
            ),
            make_simulation(
                now=3,
                doctor_used=[
                    (2, 2),
                    (3, -1),
                ],
            ),
        ]
        analysis = SimulationAnalyses(simulations)

        result = analysis.create_resource_dataframe("doctor_used")

        expected = pd.DataFrame(
            {
                0: [0, 1, 1, 2, 2, 2],
                1: [0, 0, 2, 1, 1, 1],
            },
            index=pd.RangeIndex(start=0, stop=6, step=1, name="Time"),
        )

        assert_frame_equal(result, expected)

    def test_forward_fills_between_resource_events(self):
        simulations = [
            make_simulation(
                now=4,
                nurse_used=[
                    (2, 3),
                    (4, -1),
                ],
            )
        ]
        analysis = SimulationAnalyses(simulations)

        result = analysis.create_resource_dataframe("nurse_used")

        assert result[0].tolist() == [0, 0, 3, 3, 2]

    def test_fills_minutes_before_first_event_with_zero(self):
        simulations = [
            make_simulation(
                now=5,
                doctor_used=[
                    (4, 2),
                ],
            )
        ]
        analysis = SimulationAnalyses(simulations)

        result = analysis.create_resource_dataframe("doctor_used")

        assert result[0].tolist() == [0, 0, 0, 0, 2, 2]

    def test_uses_maximum_runtime_for_all_replications(self):
        simulations = [
            make_simulation(now=2, doctor_used=[(1, 1)]),
            make_simulation(now=5, doctor_used=[(3, 2)]),
        ]
        analysis = SimulationAnalyses(simulations)

        result = analysis.create_resource_dataframe("doctor_used")

        assert result.index.tolist() == [0, 1, 2, 3, 4, 5]
        assert result.shape == (6, 2)


class TestPlotSingleSimulation:
    def test_returns_matplotlib_figure(self):
        simulation = make_simulation(
            now=120,
            doctor_used=[(30, 1), (90, -1)],
        )
        analysis = SimulationAnalyses(simulation)
        dataframe = analysis.create_resource_dataframe("doctor_used")

        figure = analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
        )

        assert isinstance(figure, matplotlib.figure.Figure)

    def test_plots_single_simulation_as_step_line(self):
        simulation = make_simulation(
            now=120,
            doctor_used=[(30, 1), (90, -1)],
        )
        analysis = SimulationAnalyses(simulation)
        dataframe = analysis.create_resource_dataframe("doctor_used")

        figure = analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
        )

        axis = figure.axes[0]
        line = axis.lines[0]

        assert line.get_label() == "Doctors"
        assert line.get_drawstyle() == "steps-post"
        assert line.get_linewidth() == 3
        assert list(line.get_xdata()) == [0, 30, 90]
        assert list(line.get_ydata()) == [0, 1, 0]

    def test_plots_multiple_resource_types(self):
        simulation = make_simulation(
            now=60,
            doctor_used=[(10, 1)],
            nurse_used=[(20, 2)],
        )
        analysis = SimulationAnalyses(simulation)

        doctors = analysis.create_resource_dataframe("doctor_used")
        nurses = analysis.create_resource_dataframe("nurse_used")

        figure = analysis.plot_resource_usage(
            [doctors, nurses],
            labels=["Doctors", "Nurses"],
        )

        axis = figure.axes[0]

        assert len(axis.lines) == 2
        assert [line.get_label() for line in axis.lines] == [
            "Doctors",
            "Nurses",
        ]

    def test_generates_default_labels(self):
        simulation = make_simulation(
            now=60,
            doctor_used=[(10, 1)],
            nurse_used=[(20, 2)],
        )
        analysis = SimulationAnalyses(simulation)

        doctors = analysis.create_resource_dataframe("doctor_used")
        nurses = analysis.create_resource_dataframe("nurse_used")

        figure = analysis.plot_resource_usage([doctors, nurses])

        axis = figure.axes[0]

        assert [line.get_label() for line in axis.lines] == [
            "Resource 1",
            "Resource 2",
        ]

    def test_sets_axis_labels_and_hour_ticks(self):
        simulation = make_simulation(
            now=121,
            doctor_used=[(10, 1)],
        )
        analysis = SimulationAnalyses(simulation)
        dataframe = analysis.create_resource_dataframe("doctor_used")

        figure = analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
        )

        axis = figure.axes[0]

        assert axis.get_xlabel() == "Hour"
        assert axis.get_ylabel() == "Resource Used"
        assert axis.get_xticks().tolist() == [0, 60, 120, 180]
        assert [label.get_text() for label in axis.get_xticklabels()] == [
            "0",
            "1",
            "2",
            "3",
        ]


class TestPlotMultipleSimulations:
    def test_plots_median_for_multiple_replications(self):
        simulations = [
            make_simulation(now=2, doctor_used=[]),
            make_simulation(now=2, doctor_used=[]),
            make_simulation(now=2, doctor_used=[]),
        ]
        analysis = SimulationAnalyses(simulations)

        dataframe = pd.DataFrame(
            {
                0: [0, 1, 2],
                1: [0, 3, 4],
                2: [0, 5, 6],
            },
            index=[0, 1, 2],
        )

        figure = analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
        )

        axis = figure.axes[0]
        median_line = axis.lines[0]

        assert list(median_line.get_xdata()) == [0, 1, 2]
        assert list(median_line.get_ydata()) == [0, 3, 4]
        assert median_line.get_label() == "Doctors"

    def test_adds_percentile_interval_for_each_resource(self):
        simulations = [
            make_simulation(now=2, doctor_used=[], nurse_used=[]),
            make_simulation(now=2, doctor_used=[], nurse_used=[]),
        ]
        analysis = SimulationAnalyses(simulations)

        doctors = pd.DataFrame(
            {
                0: [0, 1, 2],
                1: [0, 3, 4],
            }
        )
        nurses = pd.DataFrame(
            {
                0: [0, 2, 3],
                1: [0, 4, 5],
            }
        )

        figure = analysis.plot_resource_usage(
            [doctors, nurses],
            labels=["Doctors", "Nurses"],
        )

        axis = figure.axes[0]

        # Each fill_between call creates one PolyCollection.
        assert len(axis.collections) == 2
        assert len(axis.lines) == 2

    def test_uses_expected_quantiles_for_interval(self, monkeypatch):
        simulations = [
            make_simulation(now=2, doctor_used=[]),
            make_simulation(now=2, doctor_used=[]),
        ]
        analysis = SimulationAnalyses(simulations)

        dataframe = pd.DataFrame(
            {
                0: [0, 10, 20],
                1: [0, 20, 40],
                2: [0, 30, 60],
                3: [0, 40, 80],
            }
        )

        captured = {}

        original_fill_between = matplotlib.axes.Axes.fill_between

        def capture_fill_between(axis, *args, **kwargs):
            captured["x"] = list(kwargs["x"])
            captured["y1"] = list(kwargs["y1"])
            captured["y2"] = list(kwargs["y2"])
            return original_fill_between(axis, *args, **kwargs)

        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "fill_between",
            capture_fill_between,
        )

        analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
        )

        assert captured["x"] == [0, 1, 2]
        assert captured["y1"] == pytest.approx(
            dataframe.quantile(0.025, axis=1).tolist()
        )
        assert captured["y2"] == pytest.approx(
            dataframe.quantile(0.975, axis=1).tolist()
        )


class TestSavingPlots:
    def test_saves_plot_as_png_at_600_dpi(self, monkeypatch):
        simulation = make_simulation(
            now=60,
            doctor_used=[(10, 1)],
        )
        analysis = SimulationAnalyses(simulation)
        dataframe = analysis.create_resource_dataframe("doctor_used")

        savefig = monkeypatch.spy(plt, "savefig")

        analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
            save_filename="resource-usage",
        )

        savefig.assert_called_once_with(
            "resource-usage.png",
            dpi=600,
        )

    def test_does_not_save_when_filename_is_not_supplied(self, monkeypatch):
        simulation = make_simulation(
            now=60,
            doctor_used=[(10, 1)],
        )
        analysis = SimulationAnalyses(simulation)
        dataframe = analysis.create_resource_dataframe("doctor_used")

        savefig = monkeypatch.spy(plt, "savefig")

        analysis.plot_resource_usage(
            [dataframe],
            labels=["Doctors"],
        )

        savefig.assert_not_called()
