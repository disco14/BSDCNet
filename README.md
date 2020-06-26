# Bidirectional Stereo Matching Real-time Network with Double Cost Volumes

Copyright (C) 2020, Xiaogang Jia

This distribution provides an implementation, along with trained models, for the method described in our paper:

Xiaogang Jia, Bidirectional Stereo Matching Network with Double Cost Volumes.

If you find the code useful for your research, we request that you cite the above paper. Please contact jiaxiaogang@nudt.edu.cn with any questions.

## Setup

You will need a machine with a GPU and a modern version of Tensorflow with GPU support installed. You will need to compile our custom Tensorflow ops in the `slib/` directory and `corr/` directory. You can try running the `make.sh` or `compile.sh` file provided, which might work for you out of the box. If it doesn't, please look at the instructions at [https://www.tensorflow.org/guide/extend/op](https://www.tensorflow.org/guide/extend/op).

Also note that the python code assumes that your library will be compiled as `slib.so` and `corr.so`, which is the case on Linux systems. On other systems, the loadable library may have a different extension. If that is the case, please modify the `sopath` definition at the top of `slib/__init__.py` and `corr1d.py` accordingly.

## Evaluating with Trained Models

The repository contains pre-trained models in the `wts` directory for stereo matching. Please call `run.py` as follows:

```
./run.py wts/bn600k.npz data_dir
```

This code looks will look for color image files `*_10.png` corresponding to stereo pairs in the `data_dir/left` and `data_dir/right` directories, and output disparity maps as 16-bit PNG files in `data_dir/est`. You can modify the `run.py` file if you have a different directory structure / file-names.

## Training

-TODO (prepare data according to files in `data/`, and run `ptrain.py`, then `train.py` and then `popstats.py` to compute population stats for BN layers)
