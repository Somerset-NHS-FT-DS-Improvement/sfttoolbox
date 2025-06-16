import os
import unittest

import geopandas as gpd
import networkx as nx
from sfttoolbox.mapping import IsochroneGenerator
from shapely.geometry import Polygon


class TestIsochroneGenerator(unittest.TestCase):
    """
    Unit tests for the IsochroneGenerator class from the sfttoolbox.mapping module.
    This test suite verifies that graphs are correctly loaded, boundaries and isochrones
    are properly generated, data can be converted to GeoDataFrames, shortest paths
    can be calculated, and data can be save correctly.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by initialising the IsochroneGenerator instance,
        loading the road network graph for Yeovil, generating the boundary,
        and creating a test isochrone.
        """
        cls.iso_gen = IsochroneGenerator()

        # Load road network graph
        cls.iso_gen.load_graph("Yeovil", lat=50.9448, lon=-2.6343, distance=5000)
        cls.iso_gen.generate_boundary("Yeovil")

        # Generate isochrone
        cls.iso_gen.generate_isochrone(
            place_name="Yeovil",
            isochrone_name="TestIsochrone",
            lat=50.9448,
            lon=-2.6343,
            drive_time=5,
        )

    def test_graph_loaded(self):
        """
        Test that the road network graph has been successfully loaded and contains nodes and edges.
        """
        self.assertTrue(isinstance(self.iso_gen.G, nx.MultiDiGraph))
        self.assertGreater(len(self.iso_gen.G.nodes), 0)
        self.assertGreater(len(self.iso_gen.G.edges), 0)

    def test_boundary_generated(self):
        """
        Test that the boundary for the place 'Yeovil' has been generated and is a valid Polygon.
        """
        boundary = self.iso_gen.place_boundary.get("Yeovil")
        self.assertIsNotNone(boundary)
        self.assertTrue(isinstance(boundary, Polygon))

    def test_isochrone_generated(self):
        """
        Test that the isochrone named 'TestIsochrone' has been created and contains a valid Polygon.
        """
        isochrone = self.iso_gen.registry.get("TestIsochrone")
        self.assertIsNotNone(isochrone)
        self.assertTrue(isinstance(isochrone.polygon, Polygon))

    def test_convert_to_gdf(self):
        """
        Test that the road network can be successfully converted into GeoDataFrames
        containing nodes and edges.
        """
        nodes_gdf, edges_gdf = self.iso_gen.convert_road_network_to_gdf("TestIsochrone")
        self.assertIsInstance(nodes_gdf, gpd.GeoDataFrame)
        self.assertIsInstance(edges_gdf, gpd.GeoDataFrame)
        self.assertFalse(nodes_gdf.empty)
        self.assertFalse(edges_gdf.empty)

    def test_shortest_paths(self):
        """
        Test that shortest paths can be generated from the isochrone data.
        """
        paths = self.iso_gen.generate_shortest_paths("TestIsochrone")
        self.assertIsInstance(paths, list)
        self.assertGreaterEqual(len(paths), 0)

    def test_save_and_load_geojson(self):
        """
        Test that all generated data can be successfully saved to a GeoJSON file
        and that the file is created on disk.
        """
        filename = "test_isochrone_data.geojson"
        self.iso_gen.save_all_data(filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_save_graph(self):
        """
        Test that the road network graph can be saved as a GraphML file
        and that the file is created on disk.
        """
        filename = "test_graph.graphml"
        self.iso_gen.save_graph(self.iso_gen.G, filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)  # Clean up after test


if __name__ == "__main__":
    unittest.main()
