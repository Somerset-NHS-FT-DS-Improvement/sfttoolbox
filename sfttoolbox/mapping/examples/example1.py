"""
Example 1 — Isochrone Generation for Somerset and Dorset
This example demonstrates how to generate 10‑minute drive‑time isochrones
from two hospital locations, compute shortest paths, and visualise the
results on an interactive Folium map.
"""

import json

import folium
from shapely.geometry import shape

from sfttoolbox.mapping import IsochroneGenerator

# ----------------------------
# 1. Initialise generator
# ----------------------------
gen = IsochroneGenerator()

# ----------------------------
# 2. Load road networks
# ----------------------------
gen.load_graph(
    place_name="Yeovil",
    lat=50.9448,
    lon=-2.6343,
    distance=24140,  # 15 miles
)

gen.load_graph(
    place_name="Taunton",
    lat=51.0113,
    lon=-3.1207,
    distance=24140,
)

# ----------------------------
# 3. Administrative boundaries
# ----------------------------
gen.generate_boundary("Somersetshire, UK")
gen.generate_boundary("Dorset, UK")

# ----------------------------
# 4. Isochrones (10 minutes)
# ----------------------------
gen.generate_isochrone(
    place_name="Taunton",
    isochrone_name="Musgrove Park Hospital",
    lat=51.0113,
    lon=-3.1207,
    drive_time=10,
)

gen.generate_isochrone(
    place_name="Yeovil",
    isochrone_name="Yeovil Hospital",
    lat=50.9448,
    lon=-2.6343,
    drive_time=10,
)

# ----------------------------
# 5. Shortest paths & full GDF
# ----------------------------
gen.generate_shortest_paths("Musgrove Park Hospital")
gen.convert_road_network_to_gdf("Yeovil Hospital")

# ----------------------------
# 6. Save GeoJSON
# ----------------------------
out_path = "new_isochrone_data.geojson"
gen.save_all_data(out_path)

# ----------------------------
# 7. Build interactive map
# ----------------------------
with open(out_path) as f:
    geojson_data = json.load(f)

# Use centroid of first feature as map centre
first_geom = shape(geojson_data["features"][0]["geometry"])
m = folium.Map(location=[first_geom.centroid.y, first_geom.centroid.x], zoom_start=9)

# Add each feature
for feature in geojson_data["features"]:
    props = feature.get("properties", {})
    feature_type = props.get("type", "unknown")
    name = props.get("name", "Unnamed layer")

    # colour scheme preserved
    color = "red" if feature_type == "isochrone" else "black"

    fg = folium.FeatureGroup(name=name)
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
            [props["lat"], props["lon"]],
            icon=folium.Icon(icon="hospital", prefix="fa"),
        ).add_to(fg)

    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save("isochrone_map.html")
