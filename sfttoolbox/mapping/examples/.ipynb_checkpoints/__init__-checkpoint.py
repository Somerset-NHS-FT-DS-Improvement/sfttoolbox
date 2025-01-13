import json
import os
from typing import List, Union
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon, mapping
from isochrone_generator import IsochroneGenerator

# Initialize the isochrone generator
generator = IsochroneGenerator()

# Example 1: Load a graph from a city name
place_name = 'Somerset, UK'
print(f"Loading graph for {place_name}...")
generator.load_graph(place_name=place_name)
print("Graph loaded.")

# Save the city boundary as a GeoDataFrame
print("Generating city boundary...")
city_boundary = generator.generate_city_boundary(place_name)
print("City boundary generated.")
print(city_boundary)

# Example 2: Generate an isochrone
latitude = 51.2116203
longitude = -2.9830349
max_drive_time = 30  # in minutes
print(f"Generating isochrone for location ({latitude}, {longitude}) with max drive time of {max_drive_time} minutes...")
isochrone = generator.generate_isochrone(latitude, longitude, max_drive_time)
print("Isochrone generated.")
print(isochrone)

# Example 3: Generate shortest paths
print("Generating shortest paths...")
shortest_paths = generator.generate_shortest_paths()
print("Shortest paths generated.")
print(shortest_paths)

# Example 4: Convert graph to GeoJSON
print("Converting graph to GeoJSON format...")
geojson = generator.graph_to_geojson()
print("Graph converted to GeoJSON.")
print(geojson)

