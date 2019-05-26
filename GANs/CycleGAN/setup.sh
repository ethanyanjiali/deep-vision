#!/usr/bin/env bash
python -m venv env
source env/bin/activate
pip install -r requirements_gpu.in
mkdir datasets
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
unzip horse2zebra.zip
rm horse2zebra.zip
mv horse2zebra ./datasets/
mkdir -p tfrecords/horse2zebra
python tfrecords.py --dataset=horse2zebra