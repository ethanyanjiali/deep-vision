# Stacked Hourglass Network - TensorFlow 2
TensorFlow 2 implemetation of Stacked Hourglass Network (Hourglass-104) with Multi-GPU training support. This is the minimal implementation to start training. I haven't add sufficient data preprocessing and postprocessing yet, such as person cropping and NMS.


## Set Up Dataset

### MPII
Please follow [DATASET.md](../../Datasets/MPII/DATASET.md) in MPII directory to set up your dataset first. You could also use your own dataset, but this training scripts uses TF Records as data source, so you will need to generate TF Records in a similiar fashion.

## Start Training
Once all TF Records are generated, you could start training by:
```
python3 train.py
```