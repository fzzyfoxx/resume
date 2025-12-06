## app_source

### Overview
The `app_source` module contains the source code for the PerigonScout web application. The backend part is fully available within this repository, while the frontend part contains only some example sidebar components.

**technological stack** <br>
Flask | React | Leaflet | Redis | werkzeug

## Backend

The provided backend architecture is a Flask-based application designed to handle various API endpoints for managing states, filters, queries, and search areas. Below is a detailed breakdown of the architecture and its components:

#### 1. Overview

The backend is built using the Flask framework and includes the following key features:

- **Session Management**: Utilizes Flask-Session with Redis as the session store.
- **Blueprints**: Modularized routes for better organization and scalability.
- **Configuration Management**: Centralized configuration using environment variables and a `config.py` file.
- **Redis Integration**: Provides caching and session storage.
- **BigQuery Integration**: Handles query execution and result retrieval (in production mode).
- **Development Mode**: Includes mock data and artificial results for local development.

#### 2. Key Components

##### 2.1 Configuration

- **File**: `config.py`
- **Purpose**: Stores application-wide constants and configurations, such as:
    - `DEBUG` and `DEV` flags for environment differentiation.
    - Redis connection details (`REDIS_URL`, `REDIS_HOST`, `REDIS_PORT`).
    - Search-related parameters (`SEARCH_FILTER_HINTS_LIMIT`, `SEARCH_FILTER_HINTS_THRESHOLD`).
    - Table paths for results, filters, and parcels.

##### 2.2 Application Factory

- **File**: `__init__.py`
- **Purpose**: Implements the `create_app()` function to initialize the Flask application:
    - **CORS**: Configured to allow specific origins.
    - **Session Management**: Configured to use Redis for session storage.
    - **Health Check**: Provides a `/healthz` endpoint for readiness checks.
    - **Blueprint Registration**: Registers blueprints for modular route handling.
    - **Default Route**: A root endpoint (`/`) for basic connectivity testing.

##### 2.3 Blueprints

The application is divided into modular blueprints for better organization:

###### 2.3.1 State Management

- **File**: `state.py`
- **Blueprint**: `state_bp`
- **Endpoints**:
    - `/save_state`: Saves the current state with metadata (e.g., project name, ID).
    - `/list_states`: Lists all saved states with metadata and filter summaries.
    - `/load_state`: Retrieves a saved state by project ID.
    - `/delete_state`: Deletes a saved state by project ID.

###### 2.3.2 Filters

- **File**: `filters_routes.py`
- **Blueprint**: `filters_bp`
- **Endpoints**:
    - `/get_filter_spec`: Retrieves filter specifications based on provided parameters.
    - `/get_filter_search_hints`: Provides search hints for filters based on user input.

###### 2.3.3 Queries

- **File**: `queries.py`
- **Blueprint**: `queries_bp`
- **Endpoints**:
    - `/calculate_filters`: Prepares and executes a query for calculating filters.
    - `/set_search_area`: Sets the search area based on filters.
    - `/set_search_target`: Sets the target search area and retrieves results.
    - `/check_query_status`: Checks the status of a query by its ID.
    - `/get_query_result`: Retrieves the results of a query.
    - `/download_feature_csv`: Exports a GeoJSON feature to a CSV file.

##### 2.4 Redis Integration

- **Lazy Connection**: Redis connections are established lazily using `LocalProxy` to improve performance.
- **Session Storage**: Redis is used to store session data, such as states, queries, and results.

##### 2.5 BigQuery Integration

- **Lazy Loading**: BigQuery client is lazily loaded to optimize resource usage.
- **Query Execution**: Queries are executed in production mode using BigQuery, while mock data is used in development mode.
- **Result Handling**: Query results are processed and stored in the session for further use.

##### 2.6 Development Mode

- **Mock Data**: Artificial results are generated for queries and filters in development mode.
- **Environment Flag**: The `DEV` flag in the configuration determines whether the application runs in development or production mode.

#### 3. Key Features

##### 3.1 Session Management

- Sessions are used to store temporary data, such as:
    - States (`states`)
    - Queries (`queries`)
    - Results (`results`)
- Flask-Session ensures persistence using Redis as the backend.

##### 3.2 Modular Design

- The use of blueprints allows for a clean separation of concerns:
    - `state.py` for state management.
    - `filters_routes.py` for filter-related operations.
    - `queries.py` for query execution and result handling.

##### 3.3 Search and Filter Functionality

- Provides endpoints for managing filters and search areas.
- Includes advanced search features, such as search hints and filter summaries.

##### 3.4 Query Execution

- Supports query execution in both production (BigQuery) and development (mock data) environments.
- Handles query status checks and result retrieval.

#### 4. Deployment Considerations

- **Environment Variables**:
    - `FLASK_ENV`: Determines the environment (development or production).
    - `REDIS_HOST` and `REDIS_PORT`: Configure Redis connection details.
    - `SECRET_KEY`: Ensures secure session management.
- **CORS**: Configured to allow specific origins for secure cross-origin requests.
- **Health Checks**: `/healthz` endpoint ensures the application is ready to handle requests.

## Frontend

The PerigonScout web application is a modern, interactive, and highly modular frontend designed to provide users with advanced geospatial data visualization and filtering capabilities. Built using **React** and **Material-UI**, the application integrates a dynamic map interface powered by **Leaflet**, allowing users to interact with geospatial data layers, apply filters, and manage projects. The architecture emphasizes modularity, scalability, and responsiveness, ensuring a seamless user experience across devices.

### Architecture

The frontend architecture of PerigonScout is structured around React's component-based design principles. It is organized into distinct modules, each responsible for a specific aspect of the application's functionality.

#### 1. Component-Based Design

The application is divided into reusable React components, each encapsulating specific functionality. For example:

- **MapComponent**: Handles the rendering of the Leaflet map and its layers.
- **LeftSidebarComponent**: Manages the sidebar for filters, project management, and results display.
- **InfoSidebar**: Displays detailed information about selected features.
- **FiltersSection**: Manages the creation and application of filter chains.

Components are further subdivided into smaller, focused subcomponents (e.g., `AccordionStatusButton`, `AddFilterButton`) to ensure reusability and maintainability.

#### 2. State Management

- **Local State**: Managed using React's `useState`, `useEffect`, and `useCallback` hooks.
- **Shared State**: Passed down through props or managed using custom hooks (e.g., `useFilterChains`).

Key states include:

- **Filter Chains**: Managed using the `useFilterChains` hook, which handles the dynamic addition, removal, and updating of filters.
- **Map State**: Managed via a `mapRef` reference, ensuring direct interaction with the Leaflet map instance.
- **Project State**: Tracks the current project, including saved filters, markers, and results.

#### 3. Dynamic Map Integration

The map interface is powered by **Leaflet**, a lightweight and flexible JavaScript library for interactive maps.

- **MapComponent**: Handles map initialization, tile layer switching, and feature highlighting.
- **Custom Utilities**: Functions like `addShapesToMap` and `addShapesFromQuery` dynamically render GeoJSON data on the map.

#### 4. API Integration

The frontend communicates with a backend API to fetch filter specifications, query results, and project data.

- **HTTP Requests**: Managed using Axios with centralized configuration for API endpoints.
- **Key Endpoints**:
    - `/filters/get_filter_spec`: Fetches filter specifications based on user input.
    - `/queries/get_query_result`: Retrieves GeoJSON data for rendering on the map.
    - `/state/save_state` and `/state/load_state`: Manages project saving and loading.

#### 5. Material-UI for UI/UX

The application uses **Material-UI (MUI)** for consistent and responsive design.

- **Components**: Includes `Accordion`, `Tooltip`, `IconButton`, and `Typography`.
- **Custom Styles**: Ensures a clean and professional look, with a focus on usability.

### Key Techniques and Features

#### 1. Dynamic Filter Chains

- Filters are dynamically generated based on user input and API responses.
- The `useFilterChains` hook manages the lifecycle of filter chains, including fetching new filters, updating values, and handling dependencies.
- Filters support various input types, such as numeric ranges, dropdowns, and search fields.

#### 2. Interactive Map Features

- The map supports dynamic rendering of GeoJSON data, with customizable styles for polygons, points, and lines.
- Features can be highlighted on user interaction, with details displayed in the **InfoSidebar**.
- The **TileLayerSwitcher** component allows users to toggle between different map styles (e.g., CartoDB Light, Esri World Imagery).

#### 3. Project Management

- Users can save and load project states, including applied filters, map markers, and results.
- The **SaveProjectAsModal** and **LoadProjectModal** components provide intuitive interfaces for managing projects.
- Project states are serialized and stored on the backend, enabling persistence across sessions.

#### 4. Custom Hooks

- **useFilterChains**: Manages the state and logic for filter chains, including API calls and state synchronization.
- **useDebounce**: Ensures efficient API calls by debouncing user input in search fields.

#### 5. Responsive Design

- The layout adapts to different screen sizes, ensuring usability on both desktop and mobile devices.
- Sidebars (**LeftSidebarComponent**, **InfoSidebar**) are collapsible, maximizing the map's visibility when needed.

#### 6. Error Handling and Alerts

- The **AlertBox** component provides user feedback for errors, warnings, and success messages.
- API errors are caught and displayed to the user, ensuring transparency and guiding corrective actions.

#### 7. Performance Optimization

- **Debouncing**: Used in search filters to reduce unnecessary API calls.
- **Optimized Rendering**: The map's rendering logic is optimized to handle large GeoJSON datasets efficiently.
- **Memoization**: Components are memoized using `React.memo` and `useCallback` to prevent unnecessary re-renders.

