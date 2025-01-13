import json
import os
from typing import List, Union

import alphashape
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon, mapping


class IsochroneGenerator:
    def __init__(self):
        """Initialize the IsochroneGenerator"""
        self.DEFAULT_SPEED = 48.28
        self.center_node = ""
        self.sub_graph = ""
        self.polys = ""
        self.G = ""

    def load_graph(self, graph_path=None, place_name=None, network_type="drive") -> nx.MultiDiGraph:
        """
        Load graph either from a file or from OpenStreetMap.

        Parameters:
        - graph_path (str, optional): Path to the graph file to load. Defaults to None.
        - place_name (str, optional): Name of the place to load the graph from OpenStreetMap. Defaults to None.
        - network_type (str, optional): Type of street network to load (e.g., 'drive', 'walk', 'bike'). Defaults to 'drive'.

        Returns:
        - nx.MultiDiGraph: Loaded graph object.

        Raises:
        - FileNotFoundError: If the graph file is not found at the specified path.

        Notes:
        - The graph edges will be updated with travel times based on maximum speed limits after loading.
        """

        if graph_path:
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph file not found at {graph_path}")
            self.G = ox.load_graphml(graph_path)

        elif place_name:
            self.G = ox.graph_from_place(place_name, network_type=network_type)
            self._update_graph_with_times()

            graph_name = f"{place_name.replace(' ', '_').replace(',', '')}_{network_type}.graphml"
            ox.save_graphml(self.G, graph_name)

            self._update_graph_with_times()

        return self.G

    def _update_graph_with_times(self) -> None:
        """
        Update graph edges with travel times based on maximum speed limits.
        """
        for u, v, k, data in self.G.edges(data=True, keys=True):
            if max_speed := data.get("maxspeed"):
                if isinstance(max_speed, list):
                    speed = sum(self._parse_max_speed_to_kmh(s) for s in max_speed) / len(max_speed)
                else:
                    speed = self._parse_max_speed_to_kmh(max_speed)
            else:
                speed = self.DEFAULT_SPEED

            meters_per_minute = speed * 1000 / 60  # km per hour to meters per minute
            data["time"] = data["length"] / meters_per_minute

    def _parse_max_speed_to_kmh(self, max_speed: str) -> float:
        """
        Convert a max speed string to kilometers per hour.

        Parameters:
        - max_speed (str): The max speed string (e.g., '50 km/h' or '60 mph').

        Returns:
        - float: Speed in kilometers per hour.
        """
        if max_speed.lower() == "none" or not max_speed.strip():
            return self.DEFAULT_SPEED

        conversion = 1.60934 if "mph" in max_speed else 1
        try:
            speed = int(max_speed.split()[0]) * conversion
        except (ValueError, IndexError):
            speed = self.DEFAULT_SPEED

        return speed

    def generate_city_boundary(self, city_name: str) -> gpd.GeoDataFrame:
        """
        Generate the boundary of a city by its name and return it as a GeoDataFrame.

        Parameters:
        - city_name (str): The name of the city (e.g., 'Somerset, UK').

        Returns:
        - gpd.GeoDataFrame: A GeoDataFrame representing the boundary of the city.
        """

        city_boundary = ox.geocode_to_gdf(city_name)

        return city_boundary

    def generate_isochrone(self, lat: float, lon: float, max_drive_time: float, alpha: float = 30) -> Union[Polygon, List[Polygon]]:
        """
        Generate isochrone polygons based on a specified travel time.

        Parameters:
        - lat (float): Latitude of the center point.
        - lon (float): Longitude of the center point.
        - max_drive_time (float): Maximum travel time in minutes.
        - alpha (float): Alpha value for the alpha shape algorithm. Default is 30.

        Returns:
        - Union[Polygon, List[Polygon]]: Generated isochrone polygon(s).
        """
        try:
            self.center_node = ox.distance.nearest_nodes(self.G, lon, lat)
            self.sub_graph = nx.ego_graph(self.G, self.center_node, radius=max_drive_time * 60, distance="time")

            points = [
                (data["x"], data["y"]) for _, data in self.sub_graph.nodes(data=True)
            ]

            self.polys = alphashape.alphashape(points, alpha=alpha)

            if self.polys and self.polys.geom_type == "MultiPolygon":
                alpha /= 2
                self.polys = alphashape.alphashape(points, alpha=alpha)

            return self.polys
        except Exception as e:
            print(f"Error in generate_isochrone: {e}")
            return None

    def generate_shortest_paths(self):

        geojson_paths = []
        boundary_coords = (
            list(self.polys.exterior.coords)
            if self.polys.geom_type == "Polygon"
            else []
        )

        boundary_points = [Point(lon, lat) for lon, lat in boundary_coords]

        for point in boundary_points:
            nearest_node = ox.distance.nearest_nodes(self.sub_graph, point.x, point.y)

            route = nx.shortest_path(self.sub_graph, source=self.center_node, target=nearest_node, weight="length")

            path_coords = [(self.sub_graph.nodes[n]["x"], self.sub_graph.nodes[n]["y"]) for n in route]
            path_line = LineString(path_coords)

            # Convert the path into GeoJSON format (using Shapely's geo interface)
            geojson_paths.append(
                {
                    "type": "Feature",
                    "geometry": path_line.__geo_interface__,
                }
            )

        return geojson_paths

    def graph_to_geojson(self):
        """
        Convert a NetworkX MultiDiGraph object into GeoJSON format.
        """
        geojson = {"type": "FeatureCollection", "features": []}

        # Add the edges (roads) as GeoJSON LineString features
        for u, v, data in self.sub_graph.edges(data=True):
            geometry = data.get("geometry", None)

            if geometry:
                geojson_feature = {
                    "type": "Feature",
                    "geometry": mapping(geometry),
                    "properties": {
                        "from": u,
                        "to": v,
                        "road_name": data.get("name", "Unknown road"),
                    },
                }
                geojson["features"].append(geojson_feature)

        # Add the nodes (intersections) as GeoJSON Point features
        for node, data in self.sub_graph.nodes(data=True):
            position = data.get("position", None)
            if position:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": position,  # (lat, lon)
                    },
                    "properties": {"node_id": node},
                }
                geojson["features"].append(feature)

        return geojson
