#!/bin/bash


########################################################################
# unzip frames from /data/shared_dataset/MovieNet/240P/tt???????.tar
########################################################################

# mkdir /data/shared_dataset/MovieNet/keyframes_from_240P

# for file in /data/shared_dataset/MovieNet/240P/*.tar; do
#   filename=$(basename "$file" .tar)
#   echo $filename
#   tar -xvf "$file" -C /data/shared_dataset/MovieNet/keyframes_from_240P
# done

########################################################################
# extract features
########################################################################

# for cfg in 'imagenet' 'imagenet_bbc' 'imagenet_ovsd' 'places365' 'places365_bbc' 'places365_ovsd'
# do
#     python -m tool.extract.extract_features tool/extract/config/${cfg}.yaml
#     python -m tool.extract.h5_to_pkl tool/extract/config/${cfg}.yaml
# done
