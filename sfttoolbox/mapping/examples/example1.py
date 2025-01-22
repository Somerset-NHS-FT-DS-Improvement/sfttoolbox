"""
Isochrone Generation Example 1
 
This example demonstrates the usage of the IsochroneGenerator class to generate isochrones and related outputs 
for a specified location. The process includes initializing the generator with a place name, generating boundaries, 
calculating isochrones, determining shortest paths, and exporting the road network in GeoJSON format.


Make sure you have the necessary dependencies installed, including the following:

    - isochrone_generator: Main class to generate and process isochrones.
    - osmnx: For loading and manipulating street networks.
    - networkx: For graph-related operations.
    - geopandas: For handling geospatial data.
    - shapely: For geometric operations like creating and manipulating polygons and lines.
    - alphashape: For creating alpha shapes used in isochrone generation.
    - collections: For namedtuple, which is used to store isochrone data.
    - os: For file operations such as checking file existence and loading graph files.

"""
#Example Usage:
from isochrone_generator import IsochroneGenerator

# Initialize the IsochroneGenerator with a place name ('Taunton, UK'), 
# Optional: network_type -> Specify network type (defaults is 'drive')
# Optional: default_speed -> Specify default speed (defaults is 48.28 km/h

generator = IsochroneGenerator(place_name='Taunton, UK', network_type='walk', default_speed=40)

# Generate geospatial boundary for a specific location ("Taunton, UK")
boundary = generator.generate_boundary('Taunton, UK')

# Generate isochrone for a specific place ("Musgrove Park Hospital") with given lat, lon, and max_drive_time (10 minutes)
isochrone = generator.generate_isochrone('Musgrove Park Hospital', 51.017520, -3.097539, 10)

# Generate the shortest paths from the center node to the boundary points of the isochrone
shortest_paths = generator.generate_shortest_paths('Musgrove Park Hospital')

# Generate the road network in GeoJSON format for the isochrone (Musgrove Park Hospital)
road_network = generator.generate_road_network('Musgrove Park Hospital')

