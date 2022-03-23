#!/bin/bash
MODE=upload # upload | download
PROVIDER=$1 # s3 | gs
DB=$2 # dme | ims | visualize
START_DATE=$3 #ddmmyyyy
END_DATE=$4 #ddmmyyyy
TYPE=$5 # raw | processed

echo "print args $@"

remote="$PROVIDER"'://cellenmon/'"$DB"'/'"$START_DATE"'_'"$END_DATE"'/'"$TYPE"'/'
local='./CellEnMon/datasets/'"$DB"'/'"$START_DATE"'_'"$END_DATE"'/'"$TYPE"'/'

if [ $PROVIDER == "gs" ]; then
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
    aws s3 cp $remote $local --recursive
  else
    aws s3 cp $local $remote --recursive
  fi

else
  echo "$PROVIDER Proivder is not supported !"
fi
