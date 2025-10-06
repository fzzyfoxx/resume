gcloud storage buckets create gs://perigonscout.pl \
    --location=europe-west1 \
    --uniform-bucket-level-access

gcloud storage buckets add-iam-policy-binding gs://perigonscout.pl \
    --member=allUsers \
    --role=roles/storage.objectViewer

gcloud storage buckets update gs://perigonscout.pl --web-main-page-suffix=index.html