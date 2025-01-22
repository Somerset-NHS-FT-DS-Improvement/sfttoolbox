"""
Isochrone Generation Example 2

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

#Example Code:
from isochrone_generator import IsochroneGenerator

# Initialize the IsochroneGenerator with a path to your pre-generated graph file
# Example path to a GraphML file (replace with the correct path on your system)

# Initialize the IsochroneGenerator
generator = IsochroneGenerator(graph_path='path/to/your/graphml/Somerset.graphml')

# Generate geospatial boundary for a specific location (e.g., 'Somerset, UK')
boundary = generator.generate_boundary('Somerset, UK')

# Generate isochrone for a specific place ("Bridgwater Hospital") with given lat, lon, and max_drive_time (10 minutes)
isochrone = generator.generate_isochrone('Bridgwater Hospital', 51.14683, -2.97211, 10)

# Generate the shortest paths from the center node to the boundary points of the isochrone
shortest_paths = generator.generate_shortest_paths('Bridgwater Hospital')

# Generate the road network in GeoJSON format for the isochrone (Bridgwater Hospital)
road_network = generator.generate_road_network('Bridgwater Hospital')
