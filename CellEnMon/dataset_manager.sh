#!/bin/bash

PROVIDER=$1 # aws | gcp
DB=$2 # dme | ims
START_DATE=$3 #ddmmyyyy
END_DATE=$4 #ddmmyyyy
TYPE=$5 # raw | processed

echo "print args $@"

if [ $PROVIDER == "gcp" ]; then

  gcloud config set project cellenmon
  source='gs://cell_en_mon/'"$DB"'/'"$START_DATE"'-'"$END_DATE"'/'"raw"'/'
  dest='./CellEnMon/datasets/'"$DB"'/'"$TYPE"'/'"$START_DATE"'-'"$END_DATE"
  mkdir -p $dest
  gsutil -m cp -r  $source $dest

elif [ $PROVIDEDR == "aws" ]; then
  echo "$AWS_config"  | base64 -d > ~/.aws/config
  echo "$AWS_credentials" | base64 -d > ~/.aws/credentials

else
fi
