import os
import functools
from typing import List, Union
import networkx as nx
import osmnx as ox
import json
import alphashape
import shapely
import dash_leaflet as dl
from shapely.geometry import mapping, Point, Polygon
import geopandas as gpd


class IsochroneGenerator:
    def __init__(self, graph_path=None, place_name=None):
        """
        Initialize the IsochroneGenerator with a road network graph.

        Parameters:
        - graph_path (str): Path to the road network graphml file. If None, will attempt to fetch the graph from `place_name`.
        - place_name (str): Name of the place to fetch the graph from OpenStreetMap (if `graph_path` is not provided).
        - network_type (str): The type of road network to fetch ('drive', 'walk', 'bike', 'all'). Default is 'drive'.
        - default_speed (float): Default speed to use when max speed is not available for an edge.
        """
        self.DEFAULT_SPEED = 48.28
        self.graph_path = graph_path
        self.place_name = place_name
        self.center_node = ''
        self.sub_graph = ''
        self.polys = ''
        self.network_type = 'drive'
        self.G = ''
        
    @functools.lru_cache(maxsize=128)
    def _load_graph_from_file(self, graph_path : str) -> nx.Graph:
        """
        Load the road network graph from a GraphML file.

        Returns:
        - nx.Graph: The road network graph.
        """
        self.graph_path = graph_path
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Graph file not found at {self.graph_path}")
        self.G: nx.Graph = ox.load_graphml(self.graph_path)
        self.__update_graph_with_times()
        return ox.load_graphml(self.graph_path)
        
    @functools.lru_cache(maxsize=128)
    def _load_graph_from_place(self, place_name: str) -> nx.Graph:
        """
        Load the road network graph for a given place name from OpenStreetMap.

        Parameters:
        - place_name (str): The name of the place (e.g., 'Somerset, UK').

        Returns:
        - nx.Graph: The road network graph.
        """
        graph = ox.graph_from_place(place_name, network_type=self.network_type)

        self.G: nx.Graph = graph
        self.__update_graph_with_times()

        # Save the graph to a GraphML file for future use
        '''
        self.graph_path = f"{place_name.replace(' ', '_').replace(',', '')}_{self.network_type}.graphml"
        ox.save_graphml(graph, self.graph_path)
        '''
        
        return graph

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
                speed = self.DEFAULT_SPEED

            meters_per_minute = speed * 1000 / 60  # km per hour to meters per minute       
            data['time'] = data['length'] / meters_per_minute

    def __parse_max_speed_to_kmh(self, max_speed: str) -> float:
        """
        Convert a max speed string to kilometers per hour.

        Parameters:
        - max_speed (str): The max speed string (e.g., '50 km/h' or '60 mph').

        Returns:
        - float: Speed in kilometers per hour.
        """
        if max_speed.lower() == 'none' or not max_speed.strip():
            return self.DEFAULT_SPEED

        conversion = 1.60934 if "mph" in max_speed else 1
        try:
            speed = int(max_speed.split()[0]) * conversion
        except ValueError:
            # In case the conversion fails (e.g., invalid speed format), return the default speed
            speed = self.DEFAULT_SPEED
        
        return speed if speed else self.DEFAULT_SPEED
        
    def generate_city_boundary(self, city_name: str, filename: str) -> None:
        """
        Generate the boundary of a city by its name and save it as a GeoJSON file.

        Parameters:
        - city_name (str): The name of the city (e.g., 'Somerset, UK').
        - filename (str): The filename for the GeoJSON output.
        """
        # Geocode the city name to get the boundary geometry
        gdf = ox.geocode_to_gdf(city_name)

        # Save the city boundary as a GeoJSON file
        '''
        gdf.to_file(filename, driver="GeoJSON")
        '''
        return gdf
    
    @functools.lru_cache(maxsize=128)
    def generate_isochrone(self, lat: float, lon: float, max_drive_time: float) -> Union[Polygon, List[Polygon]]:
        """
        Generate isochrone polygons based on a specified travel time.

        Parameters:
        - lat (float): Latitude of the center point.
        - lon (float): Longitude of the center point.
        - max_drive_time (float): Maximum travel time in minutes.
        - use_alphashape (bool): Whether to use alphashape to create concave hulls.

        Returns:
        - Union[Polygon, List[Polygon]]: Generated isochrone polygon(s).
        """
        # Find the nearest node to the given lat/lon
        self.center_node = ox.distance.nearest_nodes(self.G, lon, lat)
        self.sub_graph = nx.ego_graph(self.G, self.center_node, radius=max_drive_time, distance='time')
        
        # Get the nodes within the max travel time radius
        nodes_within_travel_time = [
            Point(data['x'], data['y']) 
            for index, data in self.sub_graph.nodes(data=True)
        ]

        # Define the points to be used in the alphashape
        points = [(x.x, x.y) for x in nodes_within_travel_time]

        self.polys = alphashape.alphashape(points, alpha=30)
        
        # Check if the result is a MultiPolygon
        if self.polys and self.polys.geom_type == 'MultiPolygon':
            self.polys = alphashape.alphashape(points, alpha=20)

        #Better approach but time consuming plus
        #stretches the boundary too much that it misses some roads
        '''
        alpha_value = 0
        # Iterate over increasing alpha values
        while alpha_value <= 100:
            self.polys = alphashape.alphashape(points, alpha=alpha_value)
            
            # Check if the result is a MultiPolygon
            if self.polys and self.polys.geom_type == 'MultiPolygon':
                self.polys = alphashape.alphashape(points, alpha=alpha_value - 10)
                break
            
            # Increment alpha_value for the next iteration
            alpha_value += 10
        '''
            
        return self.polys
        
    
    def generate_shortest_paths(self):
        # Create a GeoDataFrame from the polygon (polys)
        boundary_gdf = gpd.GeoDataFrame(geometry=[self.polys], crs="EPSG:4326")
        
        # Convert to GeoJSON format for the boundary
        boundary_geojson = boundary_gdf.geometry.to_json()
        boundary = json.loads(boundary_geojson)
    
        # Get boundary coordinates (polygon exterior)
        # Check if self.polys is a MultiPolygon or a Polygon
        if self.polys.geom_type == 'MultiPolygon':
            # If it's a MultiPolygon, loop through each Polygon and get its exterior coordinates
            boundary_coords = []
            for poly in self.polys:
                boundary_coords.extend(list(poly.exterior.coords))
        else:
            # If it's a Polygon, just use its exterior coordinates
            boundary_coords = list(self.polys.exterior.coords)
    
        # Find nearest nodes for boundary coordinates
        boundary_nodes = []
        for lon, lat in boundary_coords:
            nearest_node = ox.distance.nearest_nodes(self.sub_graph, lon, lat)
            boundary_nodes.append(nearest_node)
    
        # Get latitude and longitude of boundary nodes
        #boundary_lat_lon = [(self.sub_graph.nodes[node]['y'], self.sub_graph.nodes[node]['x']) for node in boundary_nodes]
    
        # Prepare GeoJSON features for each shortest path from center_node to boundary nodes
        geojson_paths = []
        for node in boundary_nodes:
            # Find the shortest path from center_node to the boundary node
            route = nx.shortest_path(self.sub_graph, source=self.center_node, target=node, weight='length')
    
            # Convert path to coordinates (lat, lon) using the nodes of the path
            path_coords = [(self.sub_graph.nodes[n]['y'], self.sub_graph.nodes[n]['x']) for n in route]
    
            geojson_paths.append(dl.Polyline(positions=path_coords, color='green', weight=2))
    
        return geojson_paths


    def graph_to_geojson(self):
        """
        Convert a NetworkX MultiDiGraph object into GeoJSON format.
        """
        geojson = {"type": "FeatureCollection", "features": []}
    
        # Add the edges (roads) as GeoJSON LineString features
        for u, v, data in self.sub_graph.edges(data=True):
            # Assuming that each edge has a 'geometry' attribute with coordinates for the road
            geometry = data.get("geometry", None)  # This should be a LineString or MultiLineString
            
            if geometry:
                # Use shapely's mapping function to convert LineString to GeoJSON format
                geojson_feature = {
                    "type": "Feature",
                    "geometry": mapping(geometry),  # Direct conversion using mapping from shapely
                    "properties": {
                        "from": u,
                        "to": v,
                        "road_name": data.get("name", "Unknown road")  # Optionally, add more edge data
                    }
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
                        "coordinates": position  # (lat, lon)
                    },
                    "properties": {
                        "node_id": node
                    }
                }
                geojson["features"].append(feature)
        
        return geojson