# mxnet-reinspect (outdated)
1. MxNet would use the caffeOp for different Pooling operation.
2. The layer Transpose is implemented in https://github.com/NoListen/apollocaffe/tree/lstm_caffe (For convinience)

# Adaption
MXNet has implemented Transpose Layer accordingly and it changed its Pooling layer to have two types.

Thus, we don't need to use the CaffeOp any more and I will update it later. I suppose that it will be updated before 15th, August.

This modification is mainly because that the ['apollocaffe version'](https://github.com/Russell91/ReInspect) is too slow and about 16 frames/s on TITAN X.

This mxnet version can achieve 35 frame/s respectively and can extract features from corresponding bounding boxes using ROIPooling.


# Notification
Due to later experiment, the slow speed of the apollocaffe version is due to the python API.

One has implemented the pure C++ version and find it can also achieve 29 frames/s.


