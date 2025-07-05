import json
import os
from typing import List, Tuple

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
from shapely.geometry import Point, Polygon

from .isochrone_generator import *

__all__ = ["LineString", "Point", "Polygon", "SomersetMap"]


class SomersetMap:
    """
    Class for generating a map of Somerset with various overlays such as isochrones, bus routes, deprivation, etc.
    """

    def __init__(
        self,
        somerset_boundary_filepath="somerset_geojson_files\somerset_boundary.geojson",
    ) -> None:
        """
        Initialise SomersetMap with base layers and data sources.


        Args:
            somerset_boundary_filepath (str, optional): Filepath to the somerset boundary. Defaults to "somerset_geojson_files\somerset_boundary.geojson".
        """
        self.somerset_map = folium.Map(location=(51.1, -3.12), zoom_start=10)

        self.ig = IsochroneGenerator()

        self.somerset_boundary = gpd.read_file(somerset_boundary_filepath)

        self.hospital = {
            "Musgrove Park Hospital": (51.0113, -3.1207),
            "Yeovil Hospital": (50.9448, -2.6343),
            "Bridgwater Community Hospital": (51.14085360901964, -2.974170545072063),
            "Burnham on Sea War Memorial Hospital": (
                51.238808283995766,
                -2.9939277761197807,
            ),
            "Chard Community Hospital": (50.87498034302077, -2.952198452035544),
            "Crewkerne Community Hospital": (50.88106143909538, -2.798288304583778),
            "Frome Community Hospital": (51.2380413382477, -2.311445023871734),
            "Minehead Community Hospital": (51.19888085156418, -3.462695643764723),
            "Shepton Mallet Community Hospital": (
                51.1906165919531,
                -2.5629966353081692,
            ),
            "South Petherton Community Hospital": (
                50.95303541112399,
                -2.7983586716761666,
            ),
            "Wellington Community Hospital": (50.97624187906184, -3.2270782289259015),
            "West Mendip Community Hospital": (51.16024240105559, -2.6988811741334575),
            "Williton Community Hospital": (51.16271743071589, -3.3233721026623186),
            "Wincanton Community Hospital": (51.05876870837557, -2.418038785500917),
            "Somerton Surgery": (51.055774099646946, -2.730676692033103),
            "Victoria Park Health and Wellbeing Hub": (
                51.13048927483262,
                -3.012210117835418,
            ),
            "Lister House Surgery": (51.04113582737272, -3.313386475267806),
        }

    def _create_isochrones(
        self,
        selected_centers: List[str],
        travel_time: int,
        filepath: str = "pre_run_isochrones/",
    ) -> dict:
        """
        Generate isochrones for selected centers.

        Args:
            selected_centers (list[str]): List of selected hospital names.
            travel_time (int): Travel time in minutes.
            filepath (str, optional): Filepath to save isochrones. Defaults to "pre_run_isochrones/".

        Returns:
            dict: Dictionary of hospital names and corresponding isochrone GeoDataFrames.
        """
        isochrone_dict = {}

        for hospital_name in selected_centers:
            isochrone_filename = (
                f"{filepath}{hospital_name}_{travel_time}_min_isochrone.geojson"
            )

            if os.path.isfile(isochrone_filename):
                isochrone_dict[hospital_name] = gpd.read_file(isochrone_filename)

            else:
                # TODO: Add check that hospital_name is in hospital
                lat, lon = self.hospital[hospital_name]
                isochrone_geom = self.ig.generate_isochrone(
                    hospital_name, lat, lon, travel_time, True
                )
                isochrone_dict[hospital_name] = gpd.GeoDataFrame(
                    index=[0], crs="epsg:4326", geometry=[isochrone_geom]
                )

        return isochrone_dict

    def add_isochrone_and_markers(
        self,
        selected_centers: List[str],
        travel_time: int,
        filepath: str = "pre_run_isochrones/",
        icon_background_colour: str = "lightblue",
        colour: str = "white",
        group_prefix: str = "",
    ) -> None:
        """
        Add isochrones and markers to the map.

        Args:
            selected_centers (list[str]): List of selected hospital names.
            travel_time (int): Travel time in minutes.
            filepath (str, optional): Filepath to save isochrones. Defaults to "pre_run_isochrones/".
            icon_background_colour (str, optional): Background color for icons. Defaults to 'lightblue'.
            colour (str, optional): Color for isochrone polygons. Defaults to 'white'.
            group_prefix (str, optional): Prefix for feature groups. Defaults to "".
        """
        group_prefix = f"{group_prefix}_" if group_prefix else ""

        isochrone_group = folium.FeatureGroup(f"{group_prefix}Isochrones").add_to(
            self.somerset_map
        )
        marker_group = folium.FeatureGroup(f"{group_prefix}Markers").add_to(
            self.somerset_map
        )

        isochrones = self._create_isochrones(selected_centers, travel_time, filepath)
        for hospital_name, isochrone_df in isochrones.items():

            isochrone_clipped = gpd.clip(isochrone_df, self.somerset_boundary)
            y, x = isochrone_clipped["geometry"].iloc[0].exterior.xy

            folium.Marker(
                location=self.hospital[hospital_name],
                popup=hospital_name,
                icon=folium.Icon(
                    color=icon_background_colour,
                    icon_color=colour,
                    icon="hospital",
                    prefix="fa",
                ),
            ).add_to(marker_group)
            folium.Polygon(
                locations=zip(x, y), color=colour, fill_color=colour, fill=True
            ).add_to(isochrone_group)

    def add_bus_routes(
        self, bus_routes_filepath: str = "somerset_geojson_files/bus_routes.json"
    ) -> None:
        """
        Add bus routes to the map.

        Args:
            bus_routes_filepath (str, optional): Filepath to the bus routes. Defaults to "somerset_geojson_files/bus_routes.json".
        """
        bus_routes_group = folium.FeatureGroup("Bus Routes").add_to(self.somerset_map)

        bus_routes = json.load(open(bus_routes_filepath, "r"))

        # TODO: All this processing is ridiculous, should fix this in the file..
        # TODO: Use colour to indicate frequency??
        for routes in bus_routes:
            for route_name, route in routes.items():
                if route_name in {"TS1", "TS2", "TS3"}:
                    # school busses...
                    continue

                # Plotly allowed nans to break the route, folium does not, this splits the route at the nans
                ind = np.unique(np.where(np.isnan(route))[0])
                split_arrs = np.split(route, ind)

                # removes the nans
                for i in range(1, len(split_arrs)):
                    split_arrs[i] = split_arrs[i][1:]

                # deletes the arrays that are empty as a result
                empty_arrays = [
                    i for i, arr in enumerate(split_arrs) if arr.shape[0] == 0
                ]
                for i in sorted(empty_arrays, reverse=True):
                    del split_arrs[i]

                # Some of the list of arrays only contained nans
                if split_arrs:
                    folium.PolyLine(
                        split_arrs, tooltip=route_name, color="orange"
                    ).add_to(bus_routes_group)

    def add_deprivation(
        self,
        geo_data: str = "somerset_geojson_files\somerset_lsoa2011.geojson",
        data_filepath: str = "somerset_geojson_files\File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv",
        columns: List[str] = [
            "LSOA code (2011)",
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)",
        ],
        key_on: str = "properties.LSOA11CD",
    ) -> None:
        """
        Add deprivation data to the map.

        Args:
            geo_data (str, optional): Filepath to the geographical data. Defaults to "somerset_geojson_files\chloropleth_compatible_somerset_lsoa2011.geojson".
            data_filepath  (str, optional): Filepath to the data. Defaults to "somerset_geojson_files\File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
            columns (list[tuple[float, float]], optional): columns to use from the data. Defaults to ['LSOA code (2011)', 'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)']
            key_on (str, optional): Name of the feature in the geographical data to merge with the first column above. Defaults to properties.LSOA11CD
        """
        self._create_chloropleth(
            geo_data,
            "Deprivation",
            "Deprivation",
            pd.read_csv(data_filepath),
            columns,
            key_on,
        ).add_to(self.somerset_map)

    def add_population(
        self,
        geo_data: str = "somerset_geojson_files\somerset_lsoa2011.geojson",
        data_filepath: str = "somerset_geojson_files\File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv",
        columns: List[str] = [
            "LSOA code (2011)",
            "Total population: mid 2015 (excluding prisoners)",
        ],
        key_on: str = "properties.LSOA11CD",
    ) -> None:
        """
        Add population data to the map.

        Args:
            geo_data (str, optional): Filepath to the geographical data. Defaults to "somerset_geojson_files\chloropleth_compatible_somerset_lsoa2011.geojson".
            data_filepath  (str, optional): Filepath to the data. Defaults to "somerset_geojson_files\File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
            columns (list[tuple[float, float]], optional): columns to use from the data. Defaults to ['LSOA code (2011)', 'Total population: mid 2015 (excluding prisoners)']
            key_on (str, optional): Name of the feature in the geographical data to merge with the first column above. Defaults to properties.LSOA11CD
        """
        self._create_chloropleth(
            geo_data,
            "Population",
            "Population",
            pd.read_csv(data_filepath),
            columns,
            key_on,
        ).add_to(self.somerset_map)

    def _create_chloropleth(
        self,
        geo_data: str,
        name: str,
        legend_name: str,
        data: str,
        columns: List[str],
        key_on: str = "properties.LSOA11CD",
        fill_color: str = "RdBu",
        bins: int = 10,
        fill_opacity: float = 0.7,
        line_opacity: float = 0.1,
    ) -> folium.Choropleth:
        """
        Creates a choropleth layer for a folium map using the provided geospatial and tabular data.

        Args:
            geo_data (str): Path to the GeoJSON file or a GeoJSON string containing the geographic boundaries.
            name (str): Name of the layer to be displayed in the map's layer control.
            legend_name (str): Title for the legend displayed on the map.
            data (str): Path to the data file (e.g., CSV) or a DataFrame containing the values to visualise.
            columns (List[str]): A list with two elements: the first is the column with geographic identifiers,
            and the second is the column with the values to visualise.
            key_on (str, optional): The GeoJSON property key to match with the data. Defaults to "properties.LSOA11CD".
            fill_color (str, optional): Colour scheme for the choropleth. Defaults to "RdBu".
            bins (int, optional): Number of bins to divide the data into. Defaults to 10.
            fill_opacity (float, optional): Opacity of the fill colour. Defaults to 0.7.
            line_opacity (float, optional): Opacity of the boundary lines. Defaults to 0.1.

        Returns:
            folium.Choropleth: A folium Choropleth object that can be added to a folium map.
        """
        return folium.Choropleth(
            geo_data=geo_data,
            name=name,
            data=data,
            columns=columns,
            key_on=key_on,
            fill_color=fill_color,
            bins=bins,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name=legend_name,
        )

    def add_somerset_boundary(self) -> None:
        """
        Add Somerset boundary to the map.
        """
        folium.GeoJson(
            self.somerset_boundary.boundary,
            style_function=lambda x: {"color": "brown", "weight": 3},
            name="Somerset Boundary",
        ).add_to(self.somerset_map)

    def add_patient_coords(self, patient_coords: List[Tuple[float, float]]) -> None:
        """
        Add patient coordinates to the map.

        Args:
            patient_coords (list[tuple[float, float]]): List of patient coordinates as tuples of (latitude, longitude).
        """
        marker_cluster = MarkerCluster(name="Referrals").add_to(self.somerset_map)

        for coord in patient_coords:
            folium.features.Circle(coord, fill=True, fill_opacity=1).add_to(
                marker_cluster
            )

    def add_layer_control(self):
        """
        Add layer control to the map.
        """
        folium.LayerControl().add_to(self.somerset_map)

    def save(self, filepath: str) -> None:
        """
        Save the map to a file.

        Args:
            filepath (str): Filepath to save the map.
        """
        self.somerset_map.save(filepath)
