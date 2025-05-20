import json
import os

import alphashape
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, mapping


class IsochroneGenerator:
    def __init__(self, default_speed=48.28):
        """
        Initialize the isochrone generator with a default speed.

        Parameters:
        - default_speed (float): Speed in km/h used if no maxspeed is available.
        """
        self.default_speed = default_speed
        self.registry = {}
        self.place_boundary = {}
        self.graphs = {}
        self.circles = {}

    def load_graph(self, place_name: str, network_type="drive") -> nx.MultiDiGraph:
        """
        Load a road network graph from OpenStreetMap and update it with travel time data.

        Parameters:
        - place_name (str): Name of the place to load the graph for.
        - network_type (str): Type of street network to load.

        Returns:
        - nx.MultiDiGraph: Graph with travel times on edges.
        """
        self.G = ox.graph_from_place(place_name, network_type=network_type)
        self.__update_graph_with_times()
        self.graphs[place_name] = self.G
        return self.G

    def __update_graph_with_times(self) -> None:
        """
        Add travel time (in minutes) to each edge in the graph based on speed limits.
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

            meters_per_minute = speed * 1000 / 60
            data["time"] = float(data["length"]) / meters_per_minute

    def __parse_max_speed_to_kmh(self, max_speed: str) -> float:
        """
        Convert a maxspeed string to km/h.

        Parameters:
        - max_speed (str): Speed string, e.g., '50 km/h' or '30 mph'.

        Returns:
        - float: Speed in km/h.
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
        Get city boundary geometry as a GeoDataFrame.

        Parameters:
        - city_name (str): Name of the city.

        Returns:
        - gpd.GeoDataFrame: City boundary geometry.
        """
        city_boundary = ox.geocode_to_gdf(city_name)["geometry"].iloc[0]
        self.place_boundary[city_name] = city_boundary
        return city_boundary

    def generate_isochrone(
        self,
        city_name: str,
        name: str,
        lon: float,
        lat: float,
        drive_time: float,
        alpha: float = 30,
    ) -> Polygon:
        """
        Create an isochrone polygon for a given time threshold from a point.

        Parameters:
        - city_name (str): The name of the city the isochrone is in.
        - name (str): Identifier for the isochrone.
        - lon (float): Longitude of the center point.
        - lat (float): Latitude of the center point.
        - drive_time (float): Travel time in minutes.
        - alpha (float): Alpha shape parameter.

        Returns:
        - Polygon: Generated isochrone geometry.
        """
        self.G = self.graphs[city_name]
        center_node = ox.distance.nearest_nodes(self.G, lon, lat)
        sub_graph = nx.ego_graph(
            self.G, center_node, radius=drive_time, distance="time"
        )

        points = [(data["x"], data["y"]) for _, data in sub_graph.nodes(data=True)]
        polys = alphashape.alphashape(points, alpha=alpha)

        if polys and polys.geom_type == "MultiPolygon":
            polys = alphashape.alphashape(points, alpha=alpha / 2)

        self.registry[name] = polys, center_node, [lat, lon], drive_time, city_name
        return polys

    def generate_shortest_paths(self, isochrone_name: str):
        """
        Generate shortest path routes from center to boundary points of the isochrone.

        Parameters:
        - isochrone_name (str): Name of the isochrone.

        Returns:
        - list[dict]: List of GeoJSON features representing the paths.
        """
        isochrone_data = self.registry[isochrone_name]
        graph = self.graphs[isochrone_data[4]]
        sub_graph = nx.ego_graph(
            graph, isochrone_data[1], radius=isochrone_data[3], distance="time"
        )

        geojson_paths = []
        boundary_coords = (
            list(isochrone_data[0].exterior.coords)
            if isochrone_data[0].geom_type == "Polygon"
            else []
        )
        boundary_points = [Point(lon, lat) for lon, lat in boundary_coords]

        for point in boundary_points:
            nearest_node = ox.distance.nearest_nodes(sub_graph, point.x, point.y)
            route = nx.shortest_path(
                sub_graph,
                source=isochrone_data[1],
                target=nearest_node,
                weight="length",
            )
            path_coords = [
                (sub_graph.nodes[n]["x"], sub_graph.nodes[n]["y"]) for n in route
            ]
            path_line = LineString(path_coords)

            geojson_paths.append(
                {
                    "type": "Feature",
                    "geometry": path_line.__geo_interface__,
                }
            )

        return geojson_paths

    def generate_road_network(self, isochrone_name: str):
        """
        Export the road network within the isochrone to GeoJSON.

        Parameters:
        - isochrone_name (str): Name of the isochrone.

        Returns:
        - dict: GeoJSON with nodes and edges.
        """
        isochrone_data = self.registry[isochrone_name]
        graph = self.graphs[isochrone_data[4]]
        sub_graph = nx.ego_graph(
            graph, isochrone_data[1], radius=isochrone_data[3], distance="time"
        )

        geojson = {"type": "FeatureCollection", "features": []}

        for u, v, data in sub_graph.edges(data=True):
            geometry = data.get("geometry")
            if geometry:
                if isinstance(geometry, str):
                    geometry = wkt.loads(geometry)
                geojson["features"].append(
                    {
                        "type": "Feature",
                        "geometry": mapping(geometry),
                        "properties": {
                            "from": u,
                            "to": v,
                            "road_name": data.get("name", "Unknown road"),
                        },
                    }
                )

        for node, data in sub_graph.nodes(data=True):
            position = data.get("position")
            if position:
                geojson["features"].append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": position,
                        },
                        "properties": {"node_id": node},
                    }
                )

        return geojson

    def save_all_data(self, filename: str):
        """
        Saves all spatial data (boundaries, isochrones, circles) to a GeoJSON file,
        and serializes associated graphs to GraphML files.

        This method compiles:
        - Place boundaries from `self.place_boundary`
        - Isochrone data from `self.registry`
        - Circle center points from `self.circles`

        into a single GeoJSON `FeatureCollection` and writes it to the specified file.
        Additionally, graphs stored in `self.graphs` are saved individually as `.graphml` files.

        Parameters:
        ----------
        filename : str
            Path to the output GeoJSON file where features will be saved.
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
                    "geometry": mapping(result[0]),
                    "properties": {
                        "type": "isochrone",
                        "name": name,
                        "center_node": result[1],
                        "lat": result[2][0],
                        "lon": result[2][1],
                        "time": result[3],
                        "city_name": result[4],
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

        for graph_name, graph in self.graphs.items():
            self.save_graph(graph, f"{graph_name}.graphml")

    def save_graph(self, graph, filename: str):
        """
        Save a network graph to a GraphML file.

        Parameters:
        - graph (nx.Graph): Graph to save.
        - filename (str): Output file path.
        """
        ox.save_graphml(graph, filename)
