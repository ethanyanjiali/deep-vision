# Pascal VOC2007

[Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) is the dataset used in VOC2007 chanllenge. The goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. Specifically, I'm focusing on object detection task which predicts the bounding box and label of each object from the twenty target classes in the test image.

## Set Up Dataset

Download the training/validation data [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar), untar it into a directory called `dataset`, and make sure the direcotry structure is like:

```
dataset
|__VOC2007
    |__Annotations
    |__JPEGImages
    |__ImageSets
    |__...
    |__...
```