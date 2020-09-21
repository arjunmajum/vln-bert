# precompute_updown_img_features.py

This script will precompute image features using [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) (i.e., Faster R-CNN pretrained on Visual Genome) from panos in the Matterport3D dataset.


## Steps

Clone the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) repo. Save the `precompute_updown_img_features.py` script in the `scripts` directory. Follow the provided instructions to build the docker image and compile the simulator code.

Clone the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) repo.

From the Matterport3DSimulator top-level directory (TLD), run the Matterport3DSimulator docker, mounting both the bottom-up-attention TLD as well as the Matterport3DSimulator TDL. For example:

```
nvidia-docker run -it --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans,readonly --volume `pwd`:/root/mount/Matterport3DSimulator --volume $BOTTOM_UP_ATTENTION_TLD:/root/mount/bottom-up-attention mattersim:9.2-devel-ubuntu18.04
```

The remaining steps are run from inside the running docker image.

The bottom-up-attention repo requires Nvidia's [NCCL library](https://github.com/NVIDIA/nccl) which is used for multi-GPU training. Clone and build this somewhere in your docker.

Set up the Caffe config file (`/root/mount/bottom-up-attention/caffe/Makefile.config`) so you can build the caffe bottom-up-attention code. Use the example provided but you will need to add your NCCL build to `INCLUDE_DIRS` and `LIBRARY_DIRS`.

Install a few extra dependencies:

```
apt-get update
apt-get install libprotobuf-dev libboost-all-dev protobuf-compiler libgoogle-glog-dev libatlas-base-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libsnappy-dev
pip3 install scikit-learn scikit-image easydict protobuf pyyaml
```

Build caffe:

```
cd /root/mount/bottom-up-attention/caffe
make -j8 && make pycaffe
```

Run the script to precompute image features. Check the script, you will need to set the number of available gpus, turn off `DRY_RUN` etc.

```
cd /root/mount/Matterport3DSimulator
python3 scripts/precompute_updown_img_features.py
```
