project_id=$(gcloud config get-value project)
service_account="geodoc-repo@${project_id}.iam.gserviceaccount.com"
repository_name="images"
location="europe-west1"

gcloud artifacts repositories create ${repository_name} \
    --repository-format=docker \
    --location=${location} \
    --description="Docker images repository for GeoDoc project" \
    --async

gcloud artifacts repositories add-iam-policy-binding ${repository_name} \
    --location=${location} \
    --member="serviceAccount:${service_account}" \
    --role=roles/artifactregistry.admin

active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
gcloud iam service-accounts add-iam-policy-binding ${service_account} \
    --member="user:${active_account}" \
    --role=roles/iam.serviceAccountTokenCreator


