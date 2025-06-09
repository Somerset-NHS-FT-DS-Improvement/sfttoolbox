"""
Isochrone Generation Example – Somerset, UK

This example demonstrates how to generate drive-time isochrones using OpenStreetMap data and the IsochroneGenerator
class from `sfttoolbox`. The script performs spatial analysis to compute areas reachable by car within a set travel
time from hospitals in Somerset and Dorset counties.

The main tasks covered in this example include:

1. Loading road network graphs for:
    - Yeovil (centered at lat=50.9448, lon=-2.6343, 15-mile radius)
    - Taunton (centered at lat=51.0113, lon=-3.1207, 15-mile radius)

2. Retrieving and storing the administrative boundaries of:
    - Somerset, UK
    - Dorset, UK

3. Generating 10-minute drive-time isochrone polygons from:
    - Musgrove Park Hospital (Taunton)
    - Yeovil Hospital (Yeovil)

4. Computing:
    - Shortest path routes from Musgrove Park Hospital to isochrone boundary points
    - The full road network within the 10-minute isochrone from Yeovil Hospital

5. Saving all results to:
    - A single GeoJSON file for visualization
    - HTML map for interactive exploration
    - GraphML files (optional, via `save_graph`) for network analysis

Dependencies:
    - geopandas
    - osmnx
    - networkx
    - alphashape
    - shapely
    - folium

Usage:
    1. Run this script using Python:
        python example1.py

    2. Open `isochrone_map.html` in your browser to view the interactive map.

This example is useful for demonstrating how to apply spatial network analysis to healthcare accessibility or
transport planning problems using open data and Python tooling.
"""

import json

import folium
from sfttoolbox.mapping import IsochroneGenerator

# Initialize the generator
iso_generator = IsochroneGenerator()

# Load road networks for specific locations
iso_generator.load_graph(
    "Yeovil", lat=50.9448, lon=-2.6343, distance=24140
)  # 15 miles, default distance is 40 miles
iso_generator.load_graph("Taunton", lat=51.0113, lon=-3.1207, distance=24140)

# Store Somerset's administrative boundary
iso_generator.generate_boundary("Somerset, UK")

# Store Dorset's administrative boundary
iso_generator.generate_boundary("Dorset, UK")

# Generate 10-minute drive-time isochrones
iso_generator.generate_isochrone(
    place_name="Taunton",
    isochrone_name="Musgrove Park Hospital",
    lat=51.0113,
    lon=-3.1207,
    drive_time=10,
)  # 10 minutes
iso_generator.generate_isochrone(
    place_name="Yeovil",
    isochrone_name="Yeovil Hospital",
    lat=50.9448,
    lon=-2.6343,
    drive_time=10,
)

# Compute shortest paths and full road network within isochrones
iso_generator.generate_shortest_paths("Musgrove Park Hospital")
iso_generator.convert_road_network_to_gdf("Yeovil Hospital")

# Save results
iso_generator.save_all_data("new_isochrone_data.geojson")

# Load GeoJSON data and initialize map
with open("new_isochrone_data.geojson") as f:
    geojson_data = json.load(f)

first_feature = geojson_data["features"][0]
center_lat, center_lon = (
    first_feature["geometry"]["coordinates"][0][0][1],
    first_feature["geometry"]["coordinates"][0][0][0],
)
m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

# Plot each feature
for index, feature in enumerate(geojson_data["features"]):
    props, geometry = feature.get("properties", {}), feature["geometry"]
    feature_type = props.get("type", "unknown")
    name = props.get("name")
    color = "red" if feature_type == "isochrone" else "black"

    fg = folium.FeatureGroup(name=name, show=True)

    folium.GeoJson(
        feature,
        style_function=lambda f, col=color: {
            "fillColor": col,
            "color": col,
            "weight": 2,
            "fillOpacity": 0.1,
        },
    ).add_to(fg)

    if feature_type == "isochrone":
        folium.Marker(
            [props["lat"], props["lon"]], icon=folium.Icon(icon="hospital", prefix="fa")
        ).add_to(fg)

    fg.add_to(m)

# Add Layer Control and save the map
folium.LayerControl(collapsed=False).add_to(m)

m.save("isochrone_map.html")
