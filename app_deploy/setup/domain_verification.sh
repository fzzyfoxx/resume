# Start a DNS transaction
gcloud dns record-sets transaction start --zone=perigonscout-pl-zone

# Add the TXT record for verification.
# REPLACE the value inside the quotes with your actual verification string.
gcloud dns record-sets transaction add "google-site-verification=kVoRmj_4g0XEpu6F3Ib0X-UVJAIxbdMmXlK1askYFkM" \
    --name="perigonscout.pl." --ttl=300 --type=TXT --zone=perigonscout-pl-zone

# Execute the transaction
gcloud dns record-sets transaction execute --zone=perigonscout-pl-zone