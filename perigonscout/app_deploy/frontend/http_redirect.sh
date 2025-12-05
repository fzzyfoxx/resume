cat > redirect-map.yaml << EOL
name: geodoc-lb-http-redirect-map
defaultUrlRedirect:
  httpsRedirect: True
  redirectResponseCode: MOVED_PERMANENTLY_DEFAULT
EOL

gcloud compute url-maps import geodoc-lb-http-redirect-map \
    --source=redirect-map.yaml \
    --global

gcloud compute target-http-proxies update geodoc-lb-url-map-target-proxy \
    --url-map geodoc-lb-http-redirect-map \
    --global