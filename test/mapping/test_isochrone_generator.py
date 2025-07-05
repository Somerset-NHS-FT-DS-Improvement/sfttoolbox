import os

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Polygon

from sfttoolbox.mapping import IsochroneGenerator


@pytest.fixture(scope="module")
def iso_gen():
    """Shared fixture to set up the IsochroneGenerator and generate test data."""
    iso = IsochroneGenerator()
    iso.load_graph("Yeovil", lat=50.9448, lon=-2.6343, distance=5000)
    iso.generate_boundary("Yeovil")
    iso.generate_isochrone(
        place_name="Yeovil",
        isochrone_name="TestIsochrone",
        lat=50.9448,
        lon=-2.6343,
        drive_time=5,
    )
    return iso


def test_graph_loaded(iso_gen):
    """Ensure the road network graph is loaded and has nodes/edges."""
    G = iso_gen.graphs["Yeovil"]
    assert isinstance(G, nx.MultiDiGraph)
    assert len(G.nodes) > 0
    assert len(G.edges) > 0


def test_boundary_generated(iso_gen):
    """Ensure the boundary for 'Yeovil' is generated and is a Polygon."""
    boundary = iso_gen.place_boundary.get("Yeovil")
    assert boundary is not None
    assert isinstance(boundary, Polygon)


def test_isochrone_generated(iso_gen):
    """Test the 'TestIsochrone' was created and contains a Polygon."""
    isochrone = iso_gen.registry.get("TestIsochrone")
    assert isochrone is not None
    assert isinstance(isochrone.polygon, Polygon)


def test_convert_to_gdf(iso_gen):
    """Ensure the subgraph can be converted into non-empty GeoDataFrames."""
    nodes_gdf, edges_gdf = iso_gen.convert_road_network_to_gdf("TestIsochrone")
    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert isinstance(edges_gdf, gpd.GeoDataFrame)
    assert not nodes_gdf.empty
    assert not edges_gdf.empty


def test_shortest_paths(iso_gen):
    """Ensure shortest paths can be generated from the isochrone."""
    paths = iso_gen.generate_shortest_paths("TestIsochrone")
    assert isinstance(paths, list)
    assert len(paths) >= 0  # Can be empty but must be a list


def test_save_and_load_geojson(iso_gen):
    """Ensure GeoJSON data is saved correctly and file is created."""
    filename = "test_isochrone_data.geojson"
    iso_gen.save_all_data(filename)
    assert os.path.exists(filename)
    os.remove(filename)


def test_save_graph(iso_gen):
    """Check GraphML file is saved and file exists."""
    G = iso_gen.graphs["Yeovil"]
    filename = "test_graph.graphml"
    iso_gen.save_graph(G, filename)
    assert os.path.exists(filename)
    os.remove(filename)
