## geodoc_loader package

### Overview
The `geodoc_loader` package provides tools for data acquisition services within the PerigonScout system. It includes modules for downloading, processing, and handling geospatial data from various sources.

**technological stack** <br>
BigQuery | firestore | google-cloud-storage | shapely | asyncio | Selenium | BeautifulSoup | PyMuPDF | requests | NumPy | Pillow | GeoPandas | pyproj | pandas | rasterio | importlib | SciPy | tqdm

### Modules

#### [area](geodoc_loader/area)
Collection of BigQuery parametrized SQL queries to extract geometries for specified areas based on TERYT codes and source tables.

---
#### [data](geodoc_loader/data)
Contains single function `download_files_from_gcp` to download files with optional randomization from specified GCS bucket and file type.

---
#### [download](geodoc_loader/download)
Set of functions that handles various download & upload operations, including GC and environment setup:
- run BigQuery queries
- download files from given urls
- create GCS buckets, BigQuery datasets and tables
- upload files to GCS buckets or directly to BigQuery tables including geospatial data (`GeoJSON`, `GeoDataFrame`, `WKT`)
- **handles queue management** for multi-worker Cloud Run jobs
- validates geospatial data inputs

---
#### [handlers](geodoc_loader/handlers)
Helps with processing geospatial data formats and conversions:
- prepares multiple shapefiles from a folder for an upload to BigQuery
- converts spatial types and perform CRS transformations

---
#### [resurces](geodoc_loader/resources)
Handles fonts loading (used in SVG parsing)

---
#### [services](geodoc_loader/services)
Contains functions for specific services:
- eJournals - crawlers to acquire pdf documents from province's eJournals websites
- grid - functions to create spatial grids over specified areas
- uldk - functions to interact with ULDK API for parcels geometries acquisition

---
**Warning !** <br>
*Be aware that this package is not fully available for public access within this repository.*