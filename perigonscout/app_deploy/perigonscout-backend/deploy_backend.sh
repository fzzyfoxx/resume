# Get current GCP project
project_id=$(gcloud config get-value project)

# --- Build, Tag, and Push Docker image locally ---
# This script MUST be run from the project root directory.

echo "Building Docker image locally..."
# Build the image using the project root as context (.) and specifying the dockerfile path (-f)
docker build -t geodoc-backend -f app_deploy/perigonscout-backend/dockerfile .

echo "Tagging image for Artifact Registry..."
docker_tag="europe-west1-docker.pkg.dev/${project_id}/geodoc-backend/geodoc-backend:latest"
docker tag geodoc-backend $docker_tag

echo "Pushing image to Artifact Registry..."
# Note: You must be authenticated. Run 'gcloud auth configure-docker europe-west1-docker.pkg.dev' once.
docker push $docker_tag


# --- Deploy to Cloud Run ---
echo "Deploying to Cloud Run..."
redis_host=$(gcloud redis instances describe geodoc-redis --region=europe-west1 --format="value(host)")

gcloud run deploy geodoc-backend \
  --image="$docker_tag" \
  --region=europe-west1 \
  --platform=managed \
  --vpc-connector=geodoc-connector \
  --service-account="geodoc-backend-sa@${project_id}.iam.gserviceaccount.com" \
  --set-env-vars="REDIS_HOST=${redis_host},REDIS_PORT=6379" \
  --set-secrets="SECRET_KEY=geodoc-backend-secret-key:latest" \
  --no-allow-unauthenticated \
  --memory=4Gi \
  --cpu=4

# --- Connect Cloud Run to the Load Balancer ---
echo "Connecting Cloud Run to the Load Balancer..."

# 1. Create a Serverless Network Endpoint Group (NEG) for the Cloud Run service.
# This command is idempotent; it will not fail if the NEG already exists.
gcloud compute network-endpoint-groups create geodoc-backend-neg \
    --region=europe-west1 \
    --network-endpoint-type=serverless \
    --cloud-run-service=geodoc-backend

# 2. Add the Serverless NEG as a backend to the backend service.
# This command is also idempotent.
gcloud compute backend-services add-backend geodoc-backend-service \
    --global \
    --network-endpoint-group=geodoc-backend-neg \
    --network-endpoint-group-region=europe-west1

gcloud compute backend-services update geodoc-backend-service \
  --global --iap=disabled

gcloud run services add-iam-policy-binding geodoc-backend \
  --region=europe-west1 \
  --member=allUsers \
  --role=roles/run.invoker

gcloud projects add-iam-policy-binding geodoc-386107 \
  --member="serviceAccount:geodoc-backend-sa@geodoc-386107.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"


echo "Backend deployment complete."