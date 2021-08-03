#!/usr/bin/env bash

mkdir -p datasets

# Downloading NFBS Dataset
if [ ! -f datasets/NFBS_Dataset.tar.gz ]; then
    wget https://fcp-indi.s3.amazonaws.com/data/Projects/RocklandSample/NFBS_Dataset.tar.gz -O datasets/NFBS_Dataset.tar.gz
fi

# # Downloading ADNI dataset
# if [ ! -f  datasets/aa605acf0f2335b9b8dfdb5c66e18f68.zip ]; then
    # wget https://doid.gin.g-node.org/10.12751/g-node.aa605a/aa605acf0f2335b9b8dfdb5c66e18f68.zip -O datasets/aa605acf0f2335b9b8dfdb5c66e18f68.zip
# fi

# Downloading CC359 dataset
if [ ! -f  datasets/Original.zip ]; then
    wget --no-check-certificate -r  "https://docs.google.com/uc?export=download&id=0BxLb0NB2MjVZTXpfRVhDaE11V2c" -O datasets/Original.zip
fi

if [ ! -f  datasets/Silver-standard-machine-learning.zip ]; then
    wget --no-check-certificate -r  "https://docs.google.com/uc?export=download&id=0BxLb0NB2MjVZZWJsbmJsRTdrdms" -O datasets/Silver-standard-machine-learning.zip
fi

# extracting all files
pushd datasets
echo "Extracting NFBS_Dataset"
mkdir -p nfbs
tar xf NFBS_Dataset.tar.gz -C nfbs

echo "Extracting cc359"
mkdir -p cc359
unzip -o Original.zip -d cc359
unzip -o Silver-standard-machine-learning.zip -d cc359

# echo "Extracting ADNI"
# mkdir -p adni
# unzip -o aa605acf0f2335b9b8dfdb5c66e18f68.zip -d adni
# pushd adni
# mkdir -p images
# unzip -o MALPEM_cross-sectional_seg138_5074.zip -d images
# mkdir -p masks
# unzip -o pincram_bin_brain_masks_5074.zip -d masks
# popd
popd

mkdir -p weights/
