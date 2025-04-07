# c:/price/taxt.py
import geopandas as gpd
import pandas as pd
import os

# Debug: List files in directories to confirm paths
print("Files in data/raw/:")
print(os.listdir("data/raw/"))

# Define the shapefile path
shapefile_path = "data/raw/taxi_zones.shp"

# Verify the file exists
if not os.path.exists(shapefile_path):
    raise FileNotFoundError(f"Shapefile not found at: {shapefile_path}")

# Load the shapefile
gdf = gpd.read_file(shapefile_path)

# Check the current coordinate reference system (CRS)
print("Original CRS:", gdf.crs)

# Set the source CRS to EPSG:2263 (NY State Plane Long Island, NAD83, feet)
gdf.set_crs(epsg=2263, inplace=True)

# Reproject to EPSG:4326 (latitude/longitude in degrees)
gdf = gdf.to_crs(epsg=4326)

# Calculate centroids in the new CRS
gdf['Longitude'] = gdf.geometry.centroid.x
gdf['Latitude'] = gdf.geometry.centroid.y

# Select relevant columns and convert to regular DataFrame
gdf = pd.DataFrame(gdf[['LocationID', 'Latitude', 'Longitude']])

# Load the original taxi_zone_lookup.csv
df_zones = pd.read_csv("data/raw/taxi_zone_lookup.csv")

# Merge with the centroid coordinates
df_zones_enriched = df_zones.merge(gdf, on='LocationID', how='left')

# Save the enriched file
df_zones_enriched.to_csv("data/raw/taxi_zone_lookup_enriched.csv", index=False)
print("Enriched taxi_zone_lookup saved to: data/raw/taxi_zone_lookup_enriched.csv")