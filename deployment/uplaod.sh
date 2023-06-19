#!/bin/bash

# Set your Google Cloud project and GCS bucket name
BUCKET_NAME="ai4wildfire/VNPAFDL/CANADA"

# Set the path to the directory containing the files
FILES_DIR="date/infer"

# Set your GEE asset path
GEE_ASSET_PATH="projects/ee-eo4wildfire/assets/VNPAFDL_CANADA"


# Loop over the files in the directory
for file in "$FILES_DIR"/*; do
  if [[ -f "$file" ]]; then
    # Extract the date from the file name
    filename=$(basename "$file")
    echo filename
    date="${filename%.*}"

    # Upload the file to Google Cloud Storage (GCS)
    gsutil cp "$file" "gs://$BUCKET_NAME/$date"

    # Upload the file from GCS to GEE and assign the date as the asset ID
    earthengine upload image --asset_id="$GEE_ASSET_PATH$date" "gs://$BUCKET_NAME/$date"
  fi
done
