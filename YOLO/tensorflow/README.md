# YOLO V3 - MSCOCO 2017 - TensorFlow 2
This directory contains all the source code needed to train a YOLO V3 object detection network using MS COCO 2017 dataset.

## Set Up Dataset
Please follow [DATASET.md](../../Datasets/MSCOCO/DATASET.md) in MSCOCO directory to set up your dataset first. You could also use your own dataset, but this training scripts uses TF Records as data source, so you will need to generate TF Records in a similiar fashion.

## Start Training
Once all TF Records are generated, you could start training by:
```
make train
```

Please note that this training script support Multi-GPU training. But if you only have one GPU, it should still work.
