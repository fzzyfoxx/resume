gcloud compute backend-buckets create geodoc-frontend-bucket \
    --gcs-bucket-name=perigonscout.pl \
    --enable-cdn \
    --cache-mode=CACHE_ALL_STATIC \
    --default-ttl=3600

gcloud compute backend-services create geodoc-backend-service \
    --global \
    --load-balancing-scheme=EXTERNAL_MANAGED

gcloud compute url-maps create geodoc-lb-url-map \
    --default-backend-bucket=geodoc-frontend-bucket

gcloud compute url-maps add-path-matcher geodoc-lb-url-map \
    --path-matcher-name=api-matcher \
    --default-backend-bucket=geodoc-frontend-bucket \
    --backend-service-path-rules='/api/*=geodoc-backend-service'

gcloud compute url-maps add-host-rule geodoc-lb-url-map \
    --hosts=api.perigonscout.pl \
    --path-matcher-name=api-matcher

gcloud compute target-https-proxies create geodoc-https-proxy \
    --ssl-certificates=geodoc-ssl-cert \
    --url-map=geodoc-lb-url-map \
    --global

gcloud compute forwarding-rules create geodoc-forwarding-rule \
    --address=geodoc-lb-ip \
    --target-https-proxy=geodoc-https-proxy \
    --global \
    --ports=443