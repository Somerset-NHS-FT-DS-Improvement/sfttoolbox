import os
from collections import namedtuple

import alphashape
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon, mapping

# namedtuple to store isochrone data
IsochroneResult = namedtuple("IsochroneResult", ["isochrone_name", "polygons", "sub_graph", "center_node"])

class IsochroneGenerator:
    def __init__(self, graph_path=None, place_name=None, network_type="drive", default_speed=48.28):
        """
        Initialise the IsochroneGenerator object.

        Parameters:
        - graph_path (str): Path to a pre-existing graph file (GraphML format).
        - place_name (str): Name of the place (e.g., 'Somerset, UK') for which to generate the graph.
        - network_type (str, optional): Type of the network ('drive', 'walk', etc.). Defaults to 'drive'.
        - default_speed (float, optional): Default speed (in km/h) to use for routes without speed limit data. Defaults to 48.28 km/h.
        """

        self.default_speed = default_speed
        self.graph_path = graph_path
        self.place_name = place_name
        self.network_type = network_type
        self.registry = {}
        self.G = None

        if self.graph_path or self.place_name:
            self.__load_graph()
        else:
            raise TypeError("You must provide either graph path or place name while creating object!")

    def __load_graph(self) -> nx.MultiDiGraph:
        """
        Load graph either from a file or from OpenStreetMap.

        Returns:
        - nx.MultiDiGraph: Loaded graph object.

        Notes:
        - The graph edges will be updated with travel times based on maximum speed limits after loading.
        """

        if self.graph_path:
            if not os.path.exists(self.graph_path):
                raise FileNotFoundError(f"Graph file not found at {self.graph_path}")
            self.G = ox.load_graphml(self.graph_path)
            self.__update_graph_with_times()

        elif self.place_name:
            try:
                self.G = ox.graph_from_place(self.place_name, network_type=self.network_type)
                self.__update_graph_with_times()
            except ValueError as e:
                raise ValueError(f"Error: {e}")

        return self.G

    def __update_graph_with_times(self) -> None:
        """
        Update graph edges with travel times based on maximum speed limits.
        """
        for u, v, k, data in self.G.edges(data=True, keys=True):
            if max_speed := data.get("maxspeed"):
                if isinstance(max_speed, list):
                    speed = sum(self.__parse_max_speed_to_kmh(s) for s in max_speed) / len(max_speed)
                else:
                    speed = self.__parse_max_speed_to_kmh(max_speed)
            else:
                speed = self.default_speed

            meters_per_minute = speed * 1000 / 60  # km per hour to meters per minute
            data["time"] = data["length"] / meters_per_minute

    def __parse_max_speed_to_kmh(self, max_speed: str) -> float:
        """
        Convert a max speed string to kilometers per hour.

        Parameters:
        - max_speed (str): The max speed string (e.g., '50 km/h' or '60 mph').

        Returns:
        - float: Speed in kilometers per hour.
        """
        if max_speed.lower() == "none" or not max_speed.strip():
            return self.default_speed

        conversion = 1.60934 if "mph" in max_speed else 1
        try:
            speed = int(max_speed.split()[0]) * conversion
        except (ValueError, IndexError):
            speed = self.default_speed

        return speed

    def generate_boundary(self, city_name: str) -> gpd.GeoDataFrame:
        """
        Generate the boundary of a city by its name and return it as a GeoDataFrame.

        Parameters:
        - city_name (str): The name of the city (e.g., 'Somerset, UK').

        Returns:
        - gpd.GeoDataFrame: A GeoDataFrame representing the boundary of the city.
        """

        city_boundary = ox.geocode_to_gdf(city_name)

        return city_boundary

    def generate_isochrone(self, name: str, lat: float, lon: float, max_drive_time: float, alpha: float = 30) -> Polygon:
        """
        Generate isochrone polygons based on a specified travel time.

        Parameters:
        - name (string): Name of the isochrone
        - lat (float): Latitude of the center point.
        - lon (float): Longitude of the center point.
        - max_drive_time (float): Maximum travel time in minutes.
        - alpha (float): Alpha value for the alpha shape algorithm. Default is 30.

        Returns:
        - Polygon: The generated isochrone polygon(s).

        Example:
        result = generate_isochrone("Musgrove Park Hospital", 51.01131, -3.12074, 20)
        """
        try:
            center_node = ox.distance.nearest_nodes(self.G, lon, lat)
            sub_graph = nx.ego_graph(self.G, center_node, radius=max_drive_time, distance="time")

        except (ValueError, TypeError) as e:
            raise Exception(f"Error: {e}")

        points = [(data["x"], data["y"]) for _, data in sub_graph.nodes(data=True)]

        polys = alphashape.alphashape(points, alpha=alpha)

        if polys and polys.geom_type == "MultiPolygon":
            alpha /= 2
            polys = alphashape.alphashape(points, alpha=alpha)

        isochrone = IsochroneResult(isochrone_name=name, polygons=polys, sub_graph=sub_graph, center_node=center_node)
        self.registry[name] = isochrone

        return isochrone.polygons

    def generate_shortest_paths(self, isochrone_name: str):
        """
        Generate shortest path routes from the center node to the boundary points of the isochrone.

        Parameters:
        - isochrone_name (str): The name of the isochrone whose shortest paths are to be generated.

        Returns:
        - List[dict]: A list of GeoJSON features representing the shortest paths from the center node
          to the boundary points of the isochrone.
        """
        isochrone_data = self.registry[isochrone_name]

        geojson_paths = []

        boundary_coords = (
            list(isochrone_data.polygons.exterior.coords)
            if isochrone_data.polygons.geom_type == "Polygon"
            else []
        )

        boundary_points = [Point(lon, lat) for lon, lat in boundary_coords]

        for point in boundary_points:
            nearest_node = ox.distance.nearest_nodes(isochrone_data.sub_graph, point.x, point.y)

            route = nx.shortest_path(isochrone_data.sub_graph, source=isochrone_data.center_node, target=nearest_node, weight="length")

            path_coords = [
                (
                    isochrone_data.sub_graph.nodes[n]["x"],
                    isochrone_data.sub_graph.nodes[n]["y"],
                )
                for n in route
            ]
            path_line = LineString(path_coords)

            # Convert the path into GeoJSON format (using Shapely's geo interface)
            geojson_paths.append(
                {
                    "type": "Feature",
                    "geometry": path_line.__geo_interface__,
                }
            )

        return geojson_paths

    def generate_road_network(self, isochrone_name: str):
        """
        Convert the road network of a given isochrone's subgraph into GeoJSON format.

        Parameters:
        - isochrone_name (str): The name of the isochrone whose road network is to be converted.

        Returns:
        - dict: A GeoJSON representation of the road network, including nodes (intersections) and edges (roads).
        """

        sub_graph = self.registry[isochrone_name].sub_graph

        geojson = {"type": "FeatureCollection", "features": []}

        # Add the edges (roads) as GeoJSON LineString features
        for u, v, data in sub_graph.edges(data=True):
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
        for node, data in sub_graph.nodes(data=True):
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

    def save_graph(self, filename: str):
        """
        Save the current graph to a GraphML file.

        Parameters:
        - filename (str): The file path and name where the graph will be saved.
        """
        ox.save_graphml(self.G, filename)
