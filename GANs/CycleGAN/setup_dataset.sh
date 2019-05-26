#!/usr/bin/env bash
mkdir datasets
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
mv horse2zebra.zip ./datasets/
unzip ./datasets/horse2zebra.zip
rm ./datasets/horse2zebra.zip