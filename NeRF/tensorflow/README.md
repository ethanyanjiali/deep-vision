# Neural Radiance Field (NeRF and D-NeRF)

This directory host the source code to replicate two papers
- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, https://arxiv.org/abs/2003.08934
- Deformable Neural Radiance Fields, https://arxiv.org/abs/2011.12948

## Setup

1. Install python dependencies
```
pip install -r requirements.in
```

2. Install COLMAP. Please visit: https://colmap.github.io/install.html

3. Install ffmpeg. Please visit: https://ffmpeg.org/download.html

## Custom Training Data

To create custom training data, we will use a script `extract_frames.py` to run multiple steps in a pipeline, including blur detection and SfM.
```
python extract_frames.py --video=./data/shiba.mp4 --output-dir=./data/shiba/ --threshold=15
```