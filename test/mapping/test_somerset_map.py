import os
import shutil
import urllib.request
import zipfile

import pytest

from sfttoolbox import mapping


@pytest.fixture(scope="module")
def setup_geojson_files():
    """
    Fixture to download and prepare the Somerset GeoJSON files required for map creation.
    """
    folder_name = "somerset_geojson_files"
    zip_url = "https://github.com/Somerset-NHS-FT-DS-Improvement/somerset_geojson_files/archive/refs/heads/main.zip"
    zip_file = "geojson.zip"

    if not os.path.exists(folder_name):
        filepath, _ = urllib.request.urlretrieve(zip_url, zip_file)
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(".")
        shutil.move("somerset_geojson_files-main", folder_name)
        os.remove(zip_file)

    yield folder_name

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)


@pytest.fixture
def somerset_map_instance():
    """
    Fixture to create a SomersetMap instance.
    """
    return mapping.SomersetMap()


def test_map_instantiation(somerset_map_instance):
    """
    Test that a SomersetMap instance is created successfully.
    """
    assert isinstance(somerset_map_instance, mapping.SomersetMap)


def test_add_deprivation(somerset_map_instance, setup_geojson_files):
    """
    Test adding deprivation layer to the map.
    """
    somerset_map_instance.add_deprivation()


def test_add_bus_routes(somerset_map_instance, setup_geojson_files):
    """
    Test adding bus routes layer to the map.
    """
    somerset_map_instance.add_bus_routes()


def test_add_somerset_boundary(somerset_map_instance):
    """
    Test adding Somerset boundary to the map.
    """
    somerset_map_instance.add_somerset_boundary()


def test_add_layer_control(somerset_map_instance):
    """
    Test adding layer control to the map.
    """
    somerset_map_instance.add_layer_control()


def test_save_map(somerset_map_instance):
    """
    Test saving the map to an HTML file.
    """
    output_file = "test_map.html"
    somerset_map_instance.save(output_file)
    assert os.path.exists(output_file)
    os.remove(output_file)
