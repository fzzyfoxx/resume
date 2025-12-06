## app_deploy

### Overview
The `app_deploy` module contains bash scripts for the PerigonScout application deployment on GCP or locally.

**technological stack** <br>
Bash | gcloud CLI | Docker CLI

#### Backend
PerigonScout's backend is based on a Flask application which is containerized using Docker and served via Google Cloud Run Service.
Choosen solution allows for easy scaling and management of the application in a cloud environment. 
The application uses session memory for continuous user experience via Redis instance hosted on GCP Memorystore.

#### Frontend
The frontend is built using React and is hosted on Google Cloud Storage as a static website with CDN enabled for optimal performance.

### Deployment components
- **Backend Docker Image**: A Docker image containing the Flask application and all its dependencies.
- **GCP Cloud Run Service**: A managed service to run the backend Docker image.
- **GCP Cloud Storage Bucket**: A storage bucket to host the React frontend as a static website.
- **CDN Configuration**: Content Delivery Network setup to serve the frontend efficiently.
- **Domain Setup**: Configuration for custom domain and SSL certificates.
- **Load Balancer**: GCP Load Balancer to route traffic between frontend and backend services.
- **Redis Instance**: GCP Memorystore instance for session management.
- **VPC Network**: Virtual Private Cloud setup for secure communication between services.

---

**Warning !** <br>
*Be aware that this package is not fully available for public access within this repository.*