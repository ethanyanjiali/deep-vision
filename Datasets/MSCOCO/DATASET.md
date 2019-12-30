# Microsoft Common Objects in Context 2017

## Download Dataset
Create a folder called dataset in the root level (where the train.py is). Inside the dataset folder, follow the instructions on this page: http://cocodataset.org/#download, download images into `./dataset/train2017`, `./dataset/val2017` and `./dataset/test2017`
```bash
mkdir val2017
mkdir train2017
mkdir test2017
gsutil -m rsync gs://images.cocodataset.org/val2017 val2017
gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
gsutil -m rsync gs://images.cocodataset.org/test2017 test2017
```
Also, download the annotations as well
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

## Creating TF Record
Copy `tfrecords.py` to your newly created `dataset` folder. Run `tfrecords.py` to generate TF Records. Make sure that all dependecies have been installed (eg. ray, PIL)
```bash
python tfrecords.py
```
Also, if you want to run the notebook for inference, remember to copy the class names file `mscoco_2017_names.txt` to `dataset` too