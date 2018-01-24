import json

import mxnet as mx
import pandas as pd
from utils import (image_to_h5_mx,load_data_mean_mx, Rect, stitch_rects)
import time
import os
from scipy.misc import imread,imsave,imresize
import random
from collections import namedtuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colormaps as cmaps

config = json.load(open("./config.json", 'r'))
grid_width = config["input"]["grid_width"]
grid_height = config["input"]["grid_height"]
img_width = config["input"]["img_width"]
img_height = config["input"]["img_height"]
ht = float(config["option"]["magnify_height"])
wt = float(config["option"]["magnify_width"])
print "magnify the height",ht,"times"
print "magnify the width",wt,"times"
threshold = float(config["option"]["threshold"])

def load_video_file(video_file, data_mean):
    cap = cv2.VideoCapture(video_file)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_file)
        cv2.waitKey(1000)
        print "Wait for the header"

    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    print "frame count", cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    x0 = config["crop"]["x0"]
    y0 = config["crop"]["y0"]
    x1 = x0+img_width
    y1 = y0+img_height
    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            raw_img = frame[y0:y1, x0:x1, :]

            res = prepocessd_image(raw_img, data_mean)
            res['frame_no'] = pos_frame
            yield res
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print "frame is not ready"
            break

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break
    cap.release()

def generate_rois(batch_idx, rect, proposals):
    global wt,ht
    left_topx = rect.cx - int(rect.width/2*wt)
    left_topy = rect.cy - int(rect.height/2)
    proposals.append([batch_idx,left_topx,min(461,left_topy),int(left_topx+rect.width*wt),int(left_topy+ht*rect.height)])
    # proposals.append([batch_idx,left_topx,min(461,left_topy),left_topx+rect.width*2,left_topy+rect.height*5])

def prepocessd_image(raw_img, data_mean):
    img_processed = image_to_h5_mx(raw_img, data_mean, image_scaling=1.0)
    return {"raw": raw_img, "image": img_processed}

def forward_test(im_dict):
    
    timeExcutor.data[:] = mx.nd.array(im_dict["image"],mx.gpu())
    timeExcutor.lstm_mem_seed[:] = mx.nd.zeros((300,250),mx.gpu())
    timeExcutor.lstm_hidden_seed[:] =  mx.nd.zeros((300,250),mx.gpu())

    timeExcutor.executor.forward()
    bbox_list = [timeExcutor.bbox[idx].asnumpy() for idx in range(max_len)]
    conf_list = [timeExcutor.conf[idx].asnumpy() for idx in range(max_len)]

    all_rects = [[[] for x in range(grid_width)] for y in range(grid_height)]
    pix_per_w = img_width/grid_width
    pix_per_h = img_height/grid_height
    for n in range(len(bbox_list)):
        for k in range(grid_height * grid_width):
            #print k,n,"k n"
            y = int(k / grid_width)
            x = int(k % grid_width)
            bbox = bbox_list[n][k]
            conf = conf_list[n][k,1].flatten()[0]
            abs_cx = pix_per_w/2 + pix_per_w*x + int(bbox[0,0,0])
            abs_cy = pix_per_h/2 + pix_per_h*y + int(bbox[1,0,0])
            w = bbox[2,0,0]
            h = bbox[3,0,0]
            #print x,y
            all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
    acc_rects = stitch_rects(all_rects)
#     print len(acc_rects)
    img = im_dict["raw"]

    imsave(config["output"]["output_dir"]+"%i.jpg" % int(im_dict["frame_no"]), img)

    proposals = []
    for rect in acc_rects:
        if rect.true_confidence < threshold:
            continue
        generate_rois(0, rect, proposals)
        cv2.rectangle(img, (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),\
            (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),color=(0,0,255),thickness=2)
    imsave("./check/%i.jpg"%int(im_dict["frame_no"]), img)
    
    fexecutor.fm[:] = timeExcutor.fm
    pos_frame = int(im_dict["frame_no"])
    res = []
    for p in proposals:
        fexecutor.rois[:] = mx.nd.array(p,mx.gpu()).reshape((1,5))
        fexecutor.executor.forward()
        prefix = np.array([p[1], p[2], p[3], p[4]])
        # prefix = np.array([pos_frame, p[1], p[2], p[3], p[4]])
        res_element = np.concatenate((prefix,fexecutor.features.asnumpy().reshape(-1)))
        res.append(res_element)
    if res == []:
        res = [0]
    return np.array(res)

pretrained_cnn = mx.nd.load(config["model"]["CNN_model"])
out = mx.symbol.load(config["model"]["model_symbol"])
arg_shapes, out_shapes, _ = out.infer_shape(data=(1,3,img_height,img_width),lstm_hidden_seed=(300,250),lstm_mem_seed=(300,250))
print out_shapes

print "###### LOADING MODEL ######"
import time
start = time.clock()

arg_names = out.list_arguments()
arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape,ctx=mx.gpu(0)) for shape in arg_shapes]))

lstm_content = pd.read_hdf(config["model"]["lstm_model"],"lstm")
param_name = list(lstm_content["param_name"])
param_ndarray = list(lstm_content["param_ndarray"])

param_dict = dict(zip(param_name,param_ndarray))
for k,v in param_dict.items():
    print k,v.shape


for name in arg_names:
    if name in ["data","lstm_mem_seed","lstm_hidden_seed"]:
        continue
    key = "arg:" + name
    if key in pretrained_cnn:
        pretrained_cnn[key].copyto(arg_dict[name])
    elif "input_value" in name:
        mx.nd.array(param_dict["input_value_weight"]).copyto(arg_dict[name])
    elif "output_gate" in name:
        mx.nd.array(param_dict["output_gate_weight"]).copyto(arg_dict[name])
    elif "input_gate" in name:
        mx.nd.array(param_dict["input_gate_weight"]).copyto(arg_dict[name])
    elif "forget_value" in name:
        mx.nd.array(param_dict["forget_gate_weight"]).copyto(arg_dict[name])
    elif name in param_name:
        mx.nd.array(param_dict[name]).copyto(arg_dict[name])
    else:
        print ("SKIP arguments %s" % name)

# mx.model.save_checkpoint("listen_mxnet_first",epoch=0,symbol=out,arg_params=arg_dict,aux_params={})

print "LOAD MODEL COST ", time.clock() - start

image_mean = load_data_mean_mx(config["data"]["idl_mean"], img_width, img_height, image_scaling=1.0)

max_len = 5

executor = out.bind(ctx=mx.gpu(), args=arg_dict)
e2eExcutor = namedtuple("e2eExcutor",['executor','data','lstm_mem_seed','lstm_hidden_seed','bbox', 'conf',"fm",'arg_dict'])

timeExcutor = e2eExcutor(executor = executor, 
                         data = arg_dict["data"], 
                         lstm_mem_seed = arg_dict["lstm_mem_seed"], 
                         lstm_hidden_seed = arg_dict["lstm_hidden_seed"],
                         bbox = executor.outputs[:max_len],
                         conf = executor.outputs[max_len:],
                         fm = executor.outputs[2*max_len],
                         arg_dict=arg_dict)

print "Dectect Done!"

rois = mx.symbol.Variable(name = "rois")
fm = mx.symbol.Variable(name =  "fm")
roi_pool = mx.symbol.ROIPooling(data = fm, rois = rois, pooled_size = (7,7), spatial_scale = 0.03125)
pool5 = mx.symbol.Pooling(data = roi_pool, pool_type="max", kernel=(7,7), stride=(1,1), name= "pool5")
print pool5.list_arguments()

rois_dict = dict(zip(["rois","fm"], [mx.nd.zeros((1,5),ctx=mx.gpu(0)),mx.nd.zeros((1,1024,15,20),ctx=mx.gpu())]))
feature_executor = pool5.bind(ctx=mx.gpu(),args=rois_dict)
feature_extractor = namedtuple("feature_extracter", ["executor","rois","fm","features"])
fexecutor = feature_extractor(executor = feature_executor,
                            rois = rois_dict["rois"],
                             fm = rois_dict["fm"],
                             features = feature_executor.outputs[0])
print "ROIS DONE"

print "Done"

video_file = config["data"]["Video"]
input_gen = load_video_file(video_file, image_mean)

def forward_deploy():
    return forward_test(input_gen.next())
