echo "Tearing down expensive GCP resources..."

# 1. Delete the Load Balancer Forwarding Rule (stops LB charges)
echo "Deleting Load Balancer forwarding rule..."
gcloud compute forwarding-rules delete geodoc-forwarding-rule --global --quiet

# 2. Delete the Cloud Run service
#echo "Deleting Cloud Run service..."
#gcloud run services delete geodoc-backend --region=europe-west1 --platform=managed --quiet

# 3. Delete the Redis instance
echo "Deleting Redis instance..."
gcloud redis instances delete geodoc-redis --region=europe-west1 --quiet

# 4. Delete the VPC Connector
echo "Deleting VPC Connector..."
gcloud compute networks vpc-access connectors delete geodoc-connector --region=europe-west1 --quiet

echo "Tear down complete. Costs are minimized."
echo "NOTE: The static IP, DNS, and storage bucket remain, but have minimal cost."