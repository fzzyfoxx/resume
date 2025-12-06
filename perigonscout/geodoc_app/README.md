## geodoc_app package

### Overview
The `geodoc_app` package supports the backend of the PerigonScout application for web frontend interaction. 
It includes modules for managing inputs options, string matching search and automatic query generation for geospatial data retrieval based on user-defined criteria.

**technological stack** <br>
Flask | BigQuery | google-cloud-storage | pyproj | shapely | levenshtein | fuzzywuzzy

### Modules

#### export
Handles search results export functionality and area calculation.

---
#### inputs
Provides input options management and validation for user-defined search criteria. It also contains string matching algorithms.

---
#### search (hidden)
Handles automatic query generation and execution for geospatial data retrieval based on user-defined criteria.

---
#### utils
Utility functions for communication with GCP.

---

**Warning !** <br>
*Be aware that this package is not fully available for public access within this repository.*