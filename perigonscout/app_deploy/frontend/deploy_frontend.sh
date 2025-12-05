#!/bin/bash

# This script should be run from the project root directory.

# Define the path to your frontend application
FRONTEND_DIR="app_source/frontend"

# Navigate into the frontend directory
cd $FRONTEND_DIR

# Install dependencies to ensure they are up-to-date
echo "Installing frontend dependencies..."
npm install

# Build the React application for production
# This command creates a 'build' directory with the static files.
# If your build command or output directory is different, adjust accordingly.
echo "Building React application..."
npm run build

# Synchronize the build output with the GCS bucket
# --delete-unmatched-destination-objects is the correct flag to delete old files.
# --recursive makes the sync recursive.
echo "Uploading build files to Google Cloud Storage..."
gcloud storage rsync ./dist gs://perigonscout.pl --delete-unmatched-destination-objects --recursive

gcloud compute url-maps set-default-service geodoc-lb-url-map \
    --default-backend-bucket=geodoc-frontend-bucket \
    --global

# Navigate back to the original directory
cd -

echo "Frontend deployment complete."
echo "Your site should now be live at https://perigonscout.pl"