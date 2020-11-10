# Installation Instructions

## Install Dependencies

1. Clone the repository:
   ```
   git clone git@github.com:arjunmajum/vln-bert.git
   ```
2. Create a conda environment:
   ```
   cd vln-bert
   conda create -n vlnbert python=3.6
   conda activate vlnbert
   ```
3. Install NVIDIA Apex by following their [quick start](https://github.com/nvidia/apex#quick-start) instructions.
4. Install additional requirements:
   ```
   pip install -r requirements.txt
   ```

## Data Preprocessing

1. The code in this repository expects a variety of configuration and data
   files to exist in the `data` directory. The easiest way to get all of the
   required configuration files is to run the following command:

   ```
   python scripts/download-auxilary-data.py
   ```

2. Next, precompute image features using
   [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
   (i.e., Faster R-CNN pretrained on Visual Genome) from panos in the
   Matterport3D dataset. Follow the steps outlined in
   [scripts/matterport3D-updown-features/README.md](scripts/matterport3D-updown-features/README.md).

   Alternatively, you can download and extract the the data from here:

   [matterport-ResNet-101-faster-rcnn-genome.lmdb [13.8 GB]](https://dl.dropbox.com/s/67k2vjgyjqel6og/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip)

## Verify the directory structure

After following the steps above the `data` directory should look like this:

```
data/
  beamsearch/
  config/
  connectivity/
  distances/
  logs/
  matterport-ResNet-101-faster-rcnn-genome.lmdb
  models/
  runs/
  task/
```
