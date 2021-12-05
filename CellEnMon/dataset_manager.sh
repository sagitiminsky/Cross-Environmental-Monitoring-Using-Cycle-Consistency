DB="dme" # dme | ims
START_DATE="01012013"
END_DATE="31122015"
TYPE="raw" # raw | processed

#gcloud auth login
gcloud config set project cellenmon


source='gs://cell_en_mon/'"$DB"'/'"$START_DATE"'-'"$END_DATE"'/'"raw"'/'
dest='./CellEnMon/datasets/'"$DB"'/'"$TYPE"'/'"$START_DATE"'-'"$END_DATE"

mkdir -p dest

gsutil -m cp -r  $source $dest