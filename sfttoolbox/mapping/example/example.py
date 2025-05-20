#!/usr/bin/env python
# coding: utf-8

"""
Isochrone Generation Example – Somerset, UK

This script demonstrates how to use the `IsochroneGenerator` class to perform
spatial analysis and routing tasks using OpenStreetMap data. Specifically, it:

1. Loads a road network graph for Somerset, UK.
2. Retrieves and stores the administrative boundary of Somerset.
3. Generates 10-minute isochrone polygons from:
   - Musgrove Park Hospital
   - Bridgwater Community Hospital
4. Computes:
   - Shortest path routes from Musgrove Park Hospital to isochrone boundary points.
   - The full road network within the isochrone from Bridgwater Community Hospital.
5. Saves all results (boundaries, isochrones, and graphs) to:
   - A single GeoJSON file for visualization
   - GraphML files for further network analysis

Requirements:
    - geopandas
    - osmnx
    - networkx
    - alphashape
    - shapely
"""

from isochrone_generator import IsochroneGenerator

# Initialize the generator
iso_generator = IsochroneGenerator()

# Load the road network for Somerset
iso_generator.load_graph("Somerset, UK")

# Store the city's administrative boundary
iso_generator.generate_boundary("Somerset, UK")

# Generate a 10-minute isochrone from Musgrove Park Hospital
MPH_polygon = iso_generator.generate_isochrone(
    city_name="Somerset, UK",
    name="Musgrove Park Hospital",
    lon=-3.1195878,
    lat=51.0120657,
    drive_time=10,
)

# Generate a 10-minute isochrone from Bridgwater Community Hospital
BWC_polygon = iso.generate_isochrone(
    city_name="Somerset, UK",
    name="Bridgwater Community Hospital",
    lon=-2.974192,
    lat=51.140994,
    drive_time=10,
)

# Generate shortest paths inside MPH_polygon
MPH_shortest_routes = iso_generator.generate_shortest_paths("Musgrove Park Hospital")

# Generate entire road network inside BWC_polygon
BWC_entire_road_network = iso_generator.generate_road_network("Musgrove Park Hospital")

# Save boundaries, isochrone polygons and network graphs
iso_generator.save_all_data("isochrone_data.geojson")
