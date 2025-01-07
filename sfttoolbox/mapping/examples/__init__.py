'''
Example 1
Using an Existing Graph File
If you already have a pre-generated road network graph in GraphML format, you can simply load the graph and generate isochrones from it.

'''
from isochrone_generator import IsochroneGenerator

# Instantiate the IsochroneGenerator with a GraphML file
generator = IsochroneGenerator(graph_path="path_to_existing_graph.graphml")

# Generate an isochrone for a specific point (latitude, longitude) with a time limit
lat, lon = 51.5074, -0.1278  # Example coordinates for London
isochrone = generator.generate_isochrone(lat, lon, max_drive_time=20, use_alphashape=False)

# Save the generated isochrone to a GeoJSON file
generator.save_isochrone_to_geojson(lat, lon, max_drive_time=20, filename="my_isochrone.geojson")

'''
Example 2
Fetching the Graph Based on Place Name
If you don't have a GraphML file and wish to fetch the road network dynamically based on a place name, you can do so with osmnx by providing the place name and network type.

'''
from isochrone_generator import IsochroneGenerator

# Instantiate the IsochroneGenerator by providing a place name and network type
generator = IsochroneGenerator(place_name="Swindon, UK", network_type="drive")

# Generate an isochrone with a 20-minute drive time from a given point
lat, lon = 51.5550, -1.7810  # Example coordinates for Swindon, UK
isochrone = generator.generate_isochrone(lat, lon, max_drive_time=20, use_alphashape=True)

# Save the generated isochrone to a GeoJSON file
generator.save_isochrone_to_geojson(lat, lon, max_drive_time=20, filename="swindon_drive_isochrone.geojson")
