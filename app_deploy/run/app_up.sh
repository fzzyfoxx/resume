echo "Bringing application up..."

# 1. Re-create the Redis instance
echo "Creating Redis instance..."
gcloud redis instances create geodoc-redis \
    --size=1 \
    --region=europe-west1 \
    --tier=BASIC \
    --network=default

# 2. Re-create the VPC Connector
echo "Creating VPC Connector..."
gcloud compute networks vpc-access connectors create geodoc-connector \
    --region=europe-west1 \
    --network=default \
    --range=10.8.0.0/28

# 3. Run your existing backend deployment script
# This will build the image, deploy to Cloud Run, and connect to the LB backend service
#echo "Deploying backend application..."
#sh /home/fzzyfoxx/projects/GeoDoc/app_deploy/perigonscout-backend/deploy_backend.sh

# 4. Re-create the Load Balancer Forwarding Rule to make the app live
echo "Creating Load Balancer forwarding rule..."
gcloud compute forwarding-rules create geodoc-forwarding-rule \
    --address=geodoc-lb-ip \
    --target-https-proxy=geodoc-https-proxy \
    --global \
    --ports=443

echo "Application is up and running."