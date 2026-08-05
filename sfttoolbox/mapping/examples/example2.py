"""
Example 2 — Somerset Deprivation & Bus Route Map

Downloads Somerset GeoJSON files (if missing), constructs a SomersetMap
with deprivation scores, bus routes, and boundaries. Output: map.html
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from sfttoolbox import mapping

# ----------------------------
# File paths
# ----------------------------
DATA_DIR = Path("somerset_geojson_files")
SOMERSET_BOUNDARY = DATA_DIR / "somerset_boundary.geojson"
LSOA_GEOJSON = f"{DATA_DIR}/somerset_lsoa2011.geojson"
DEPRIVATION_CSV = f"{DATA_DIR}/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
BUSROUTES_JSON = f"{DATA_DIR}/bus_routes.json"

# ----------------------------
# Download if missing
# ----------------------------
if not os.path.exists(DATA_DIR):
    print("GeoJSON folder not found — downloading from GitHub.")
    zip_path, _ = urllib.request.urlretrieve(
        "https://github.com/Somerset-NHS-FT-DS-Improvement/"
        "somerset_geojson_files/archive/refs/heads/main.zip"
    )
    print("Unzipping files …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
    shutil.move("somerset_geojson_files-main", DATA_DIR)

# ----------------------------
# Build the map
# ----------------------------
print("Creating Somerset map …")
sm = mapping.SomersetMap(somerset_boundary_filepath=SOMERSET_BOUNDARY)

sm.add_deprivation(
    data_filepath=DEPRIVATION_CSV,
    geo_data=LSOA_GEOJSON,
)

sm.add_bus_routes(bus_routes_filepath=BUSROUTES_JSON)
sm.add_somerset_boundary()
sm.add_layer_control()

sm.save("map.html")

print("Saved map.html")
