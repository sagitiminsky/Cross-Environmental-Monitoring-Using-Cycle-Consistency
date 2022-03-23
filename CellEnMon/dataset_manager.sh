#!/bin/bash
MODE=upload # upload | download
PROVIDER=$1 # s3 | gs
DB=$2 # dme | ims
START_DATE=$3 #ddmmyyyy
END_DATE=$4 #ddmmyyyy
TYPE=$5 # raw | processed

echo "print args $@"

remote="$PROVIDER"'://cell_en_mon/'"$DB"'/'"$START_DATE"'-'"$END_DATE"'/'"raw"'/'
local='./CellEnMon/datasets/'"$DB"'/'"$TYPE"'/'"$START_DATE"'-'"$END_DATE"

if [ $PROVIDER == "gcp" ]; then
  #google auth
  gcloud config set project cellenmon

  if [ $MODE == "download"]; then
    mkdir -p $dest
    gsutil -m cp -r  $remote $local
  else
    gsutil -m cp -r  $local $remote
  fi

elif [ $PROVIDEDR == "aws" ]; then
  #aws auth
  export AWS_ACCESS_KEY_ID=python3 CellEnMon/libs/vault/vault.py aws key
  export AWS_SECRET_ACCESS_KEY=python3 CellEnMon/libs/vault/vault.py aws secret

    if [ $MODE == "download"]; then
    mkdir -p $dest
    s3 -m cp -r  $remote $local
  else
    s3 -m cp -r  $local $remote
  fi



else
fi
