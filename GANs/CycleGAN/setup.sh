#!/usr/bin/env bash
python -m venv env
source env/bin/activate
pip install -r requirements_gpu.in
mkdir datasets
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
unzip monet2photo.zip
rm monet2photo.zip
mv monet2photo ./datasets/
mkdir -p tfrecords/monet2photo
python tfrecords.py --dataset=monet2photo