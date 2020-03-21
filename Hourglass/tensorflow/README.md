# Stacked Hourglass Network - TensorFlow 2
TensorFlow 2 implemetation of Stacked Hourglass Network (Hourglass-104) with Multi-GPU training support. I've trained the model with MPII dataset. To train with your own dataset, you will need to create TF Records and maybe change the preprocess.py file accordinly.


## Set Up Dataset

### MPII
Please follow [DATASET.md](../../Datasets/MPII/DATASET.md) in MPII directory to set up your dataset first. You could also use your own dataset, but this training scripts uses TF Records as data source, so you will need to generate TF Records in a similiar fashion.

## Start Training
Once all TF Records are generated, you could start training by:
```
python3 train.py
```

## Inference

Please see `demo_hourglass_pose.ipynb` for examples of inference.