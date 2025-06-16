"""
Isochrone Module

This module provides a framework for generating isochrone polygons based on travel times within road networks
using OpenStreetMap data. It uses the `osmnx`, `networkx`, and `alphashape` libraries to load graphs, build
subgraphs based on drive times, and generate geometric representations of reachable areas.

Class:
    - IsochroneGenerator: Manages graph loading, travel time calculations, isochrone generation,
      shortest path extraction, GeoDataFrame conversion, and export operations.

Usage:
    1. Import the module and required libraries:
       ```python
       import json
       import os
       import alphashape
       import geopandas as gpd
       import networkx as nx
       import osmnx as ox
       from collections import namedtuple
       from shapely import wkt
       ```
    2. Create an instance of `IsochroneGenerator`.
    3. Load a graph using a place name or coordinates with `load_graph`.
    4. Generate an isochrone polygon using `generate_isochrone`.
    5. Optionally:
        - Extract the boundary using `generate_boundary`.
        - Convert the graph to GeoDataFrames using `convert_road_network_to_gdf`.
        - Generate shortest paths within the isochrone using `generate_shortest_paths`.
        - Export isochrone and boundary data to GeoJSON using `save_all_data`.
        - Save network graphs using `save_graph`.

Example:
    See the file titled `example.py` in the `examples` directory.

This module is intended for geospatial and mobility analytics, especially in urban planning,
logistics, and service area visualisation.
"""

import json
import os
from collections import namedtuple

import alphashape
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely import wkt
from shapely.geometry import *

IsochroneRegistry = namedtuple(
    "IsochroneRegistry",
    ["polygon", "centre_node", "lat", "lon", "drive_time", "place_name", "sub_graph"],
)


class IsochroneGenerator:
    def __init__(self, default_speed: float = 48.28032):
        """
        Initialise the IsochroneGenerator with a default speed for edges missing speed data.

        Args:
            default_speed (float): Default travel speed in km/h. Defaults to 48.28032 km/h = 30 miles/h.
        """
        self.default_speed = default_speed
        self.registry = {}
        self.place_boundary = {}
        self.graphs = {}
        self.circles = {}  # reserved for future use

    def load_graph(
        self,
        place_name: str,
        lat: float = None,
        lon: float = None,
        distance: float = 64374,  # in case distance isn't provided, defaul distance (which is 40 miles) will be use to load the graph
        network_type: str = "drive",
    ) -> nx.MultiDiGraph:
        """
        Load a road network from OpenStreetMap, either by place name or by geographic coordinates.

        Args:
            place_name (str): Name of the place (used if lat/lon not provided).
            lat (float, optional): Latitude for location-based loading.
            lon (float, optional): Longitude for location-based loading.
            distance (float): Distance in meters around the lat/lon to load the graph. Default is 64374.
            network_type (str): Type of road network to load (e.g., "drive", "walk").

        Returns:
            nx.MultiDiGraph: A directed multigraph with time-annotated edges.

        Raises:
            ValueError: If neither place_name nor lat/lon are provided.
        """
        if lat is not None and lon is not None:
            self.G = ox.graph_from_point(
                center_point=[lat, lon],
                dist=distance,
                dist_type="network",
                network_type=network_type,
                simplify=True,
            )
        else:
            self.G = ox.graph_from_place(place_name, network_type=network_type)

        self.__update_graph_with_times()
        self.graphs[place_name] = self.G
        return self.G

    def __update_graph_with_times(self) -> None:
        """
        Annotate graph edges with estimated travel times based on edge length and speed.
        """
        for u, v, k, data in self.G.edges(data=True, keys=True):
            if max_speed := data.get("maxspeed"):
                if isinstance(max_speed, list):
                    speed = sum(
                        self.__parse_max_speed_to_kmh(s) for s in max_speed
                    ) / len(max_speed)
                else:
                    speed = self.__parse_max_speed_to_kmh(max_speed)
            else:
                speed = self.default_speed

            speed *= 0.9  # Account for real-world delays like traffic.
            meters_per_minute = speed * 1000 / 60
            data["time"] = float(data["length"]) / meters_per_minute

    def __parse_max_speed_to_kmh(self, max_speed: str) -> float:
        """
        Convert a maxspeed string to a float value in km/h.

        Args:
            max_speed (str): String representation of speed, e.g., '30 mph', '50 km/h'.

        Returns:
            float: Parsed speed in km/h.
        """
        if max_speed.lower() == "none" or not max_speed.strip():
            return self.default_speed

        conversion = 1.60934 if "mph" in max_speed else 1
        try:
            speed = int(max_speed.split()[0]) * conversion
        except (ValueError, IndexError):
            speed = self.default_speed

        return speed

    def generate_boundary(self, place_name: str) -> gpd.GeoDataFrame:
        """
        Retrieve and store the administrative boundary for a location.

        Args:
            place_name (str): Name of the place to fetch the boundary for.

        Returns:
            gpd.GeoDataFrame: Geometry of the place boundary.
        """
        place_boundary = ox.geocode_to_gdf(place_name)["geometry"].iloc[0]
        self.place_boundary[place_name] = place_boundary
        return place_boundary

    def generate_isochrone(
        self,
        place_name: str,
        isochrone_name: str,
        lat: float,
        lon: float,
        drive_time: float,
        alpha: float = 30,
    ) -> Polygon:
        """
        Generate an isochrone polygon for a given location and drive time.

        Args:
            place_name (str): Name of the place associated with the graph.
            isochrone_name (str): Identifier for the isochrone.
            lat (float): Latitude of the center point.
            lon (float): Longitude of the center point.
            drive_time (float): Time limit (in minutes) from the center node.
            alpha (float): Alpha parameter for alpha shape generation.

        Returns:
            Polygon: A shapely Polygon representing the isochrone boundary.
        """
        self.G = self.graphs[place_name]
        centre_node = ox.distance.nearest_nodes(self.G, lon, lat)
        sub_graph = nx.ego_graph(
            self.G, centre_node, radius=drive_time, distance="time"
        )
        points = [(data["x"], data["y"]) for _, data in sub_graph.nodes(data=True)]
        polygon = alphashape.alphashape(points, alpha=alpha)

        if polygon and polygon.geom_type == "MultiPolygon":
            polygon = alphashape.alphashape(points, alpha=alpha / 2)

        self.registry[isochrone_name] = IsochroneRegistry(
            polygon, centre_node, lat, lon, drive_time, place_name, sub_graph
        )
        return polygon

    def generate_shortest_paths(self, isochrone_name: str) -> list[dict]:
        """
        Generate shortest path LineStrings from the isochrone center to its polygon boundary.

        Args:
            isochrone_name (str): Name of the isochrone in the registry.

        Returns:
            list[dict]: A list of GeoJSON-like LineString features representing shortest paths.
        """
        isochrone_data = self.registry[isochrone_name]
        sub_graph = isochrone_data.sub_graph

        geojson_paths = []
        boundary_coords = (
            list(isochrone_data.polygon.exterior.coords)
            if isochrone_data.polygon.geom_type == "Polygon"
            else []
        )
        boundary_points = [Point(lon, lat) for lon, lat in boundary_coords]

        for point in boundary_points:
            nearest_node = ox.distance.nearest_nodes(sub_graph, point.x, point.y)
            route = nx.shortest_path(
                sub_graph,
                source=isochrone_data.centre_node,
                target=nearest_node,
                weight="length",
            )
            path_coords = [
                (sub_graph.nodes[n]["x"], sub_graph.nodes[n]["y"]) for n in route
            ]
            geojson_paths.append(
                {
                    "type": "Feature",
                    "geometry": LineString(path_coords).__geo_interface__,
                }
            )

        return geojson_paths

    def convert_road_network_to_gdf(
        self, isochrone_name: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Convert the subgraph associated with an isochrone to GeoDataFrames.

        Args:
            isochrone_name (str): Name of the isochrone in the registry.

        Returns:
            tuple: A tuple of (nodes GeoDataFrame, edges GeoDataFrame).
        """
        sub_graph = self.registry[isochrone_name].sub_graph
        return ox.graph_to_gdfs(sub_graph, nodes=True, edges=True)

    def save_all_data(self, filename: str) -> None:
        """
        Save all boundary, isochrone, and circle data to a GeoJSON file.

        Args:
            filename (str): Path to the output GeoJSON file.
        """
        features = []

        for name, geom in self.place_boundary.items():
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {"type": "boundary", "name": name},
                }
            )

        for name, result in self.registry.items():
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(result.polygon),
                    "properties": {
                        "type": "isochrone",
                        "name": name,
                        "centre_node": result.centre_node,
                        "lat": result.lat,
                        "lon": result.lon,
                        "time": result.drive_time,
                        "place_name": result.place_name,
                    },
                }
            )

        for name, circle in self.circles.items():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [circle[0], circle[1]],
                    },
                    "properties": {
                        "type": "circle",
                        "name": name,
                        "radius_m": circle[2],
                    },
                }
            )

        with open(filename, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

    def save_graph(self, graph: nx.MultiDiGraph, filename: str) -> None:
        """
        Save a NetworkX graph to a GraphML file.

        Args:
            graph (nx.MultiDiGraph): The road network graph to be saved.
            filename (str): Destination file path.
        """
        ox.save_graphml(graph, filename)
