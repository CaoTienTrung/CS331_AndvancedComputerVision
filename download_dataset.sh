#!/bin/bash

# Thư mục Dataset
if [ ! -d "Dataset" ]; then
    echo "Creating Dataset directory..."
    mkdir Dataset
else
    echo "Dataset directory already exists. Skipping creation."
fi


# Tải file bằng gdown
echo "Downloading file..."
gdown --id 1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S -O Dataset/myfile.zip

# Giải nén file (Windows Git Bash có unzip?)
if command -v unzip &> /dev/null
then
    echo "Extracting zip file..."
    unzip -o Dataset/myfile.zip -d Dataset
    echo "Deleting zip file..."
    rm Dataset/myfile.zip
else
    echo "unzip command not found, skipping extraction."
fi

# FSC-147-S.json
curl -L -o Dataset/FSC-147-S.json \
https://raw.githubusercontent.com/cha15yq/T2ICount/main/FSC-147-S.json

# ImageClasses_FSC147.txt
curl -L -o Dataset/ImageClasses_FSC147.txt \
https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/ImageClasses_FSC147.txt

# annotation_FSC147_384.json
curl -L -o Dataset/annotation_FSC147_384.json \
https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/annotation_FSC147_384.json

# Train_Test_Val_FSC_147.json
curl -L -o Dataset/Train_Test_Val_FSC_147.json \
https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/Train_Test_Val_FSC_147.json

echo "Done!"
