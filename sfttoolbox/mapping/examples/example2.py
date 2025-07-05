import os
import shutil
import urllib.request
import zipfile

from sfttoolbox import mapping

# Download the relevant geojson files and unzip them.
if not os.path.exists("somerset_geojson_files"):
    print("somerset_geojson_files folder not found, downloading.")
    filepath, headers = urllib.request.urlretrieve(
        "https://github.com/Somerset-NHS-FT-DS-Improvement/somerset_geojson_files/archive/refs/heads/main.zip"
    )

    print("unzipping somerset_geojson_files")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(".")

    print("renaming somerset_geojson_files-main to somerset_geojson_files")
    shutil.move("somerset_geojson_files-main", "somerset_geojson_files")

print("creating the map!")
# Instantiate the map
sm = mapping.SomersetMap()

sm.add_deprivation()
sm.add_bus_routes()
sm.add_somerset_boundary()

# Allow users to switch between layers
sm.add_layer_control()

# Other functions include add_population or _create_chloropleth can be used to generate a chloropleth layer that can then be added.

sm.save("map.html")
