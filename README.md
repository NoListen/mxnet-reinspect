# mxnet-reinspect

MXNet version of [Reinspect](https://github.com/Russell91/reinspect)

# Features
- No training included
- 35 fps for detection (Nvidia TITANX)
- enlarge the bbox to the whole body by scales varying from scenes to scenes
- ROI Pooling for bounding boxes to extract features

# Additional Requirements
- Transfer caffemodel to MXNet using caffe_converter in mxnet/tools.

# Illustration
- mxnet_track.py is the main file
- config.json describes the hyperparameters
- reinspect.json is the network architecture file
- utils is similar to that in [Reinspect](https://github.com/Russell91/reinspect) with redundant files moved.
- model-transfer specifies the needs to deal with the model learned in [Reinspect](https://github.com/Russell91/reinspect)

# Process
- mxnet model load googlenet params
- mxnet model load lstm params (lstm.h5)
- output the proposals
- extract the features of bbox proposals

# Note
- tracking is not included which you can refer to [MOT_XCODE](https://github.com/HansSJTU/MOT_XCODE).
- bridging python and C++ is not included which you can refer to [numpy-opencv-converter](https://github.com/spillai/numpy-opencv-converter).

# TODO
- add learning process to mxnet-reinspect.
