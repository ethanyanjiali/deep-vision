# VOC 2007 Dataset

## Setup the directory
Inside the directory of the training script, such as `YOLO/tensorflow/`, create a directory called `dataset`.
Then, copy `tfrecords.py` and `voc_2007_names.txt` to this `dataset`.

## Download and extract dataset
In the `dataset` directory:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar
```

## Generate TF Records
In the `dataset` directory:
```
python3 tfrecords.py
```