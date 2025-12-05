gcloud redis instances create geodoc-redis \
    --size=1 \
    --region=europe-west1 \
    --tier=BASIC \
    --network=default

gcloud redis instances describe geodoc-redis --region=europe-west1 --format="value(host)"