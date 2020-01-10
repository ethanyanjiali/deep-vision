# MPII Human Pose Dataset

In `Hourglass/tensorflow` directory (or similiar others), run:
```bash
mkdir dataset
cd dataset
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
tar -xvf mpii_human_pose_v1.tar.gz -C mpii
unzip mpii_human_pose_v1_u12_2.zip
```

However, the annotation above is in MATLAB format and hard to read. I'm using a JSON version of annotations from here:
https://github.com/microsoft/multiview-human-pose-estimation-pytorch/blob/master/INSTALL.md#data-preparation
which leads to this OneDrive: https://onedrive.live.com/?authkey=%21AMdfUcJgrZBwTRU&id=93774C670BD4F835%211101&cid=93774C670BD4F835

Once you have all the images and annotations, copy `tfrecords_mpii.py` to the new `dataset` folder and run it to generate TF Records. In order to crop person, you also need to modify the script to add `center` and `scale`.