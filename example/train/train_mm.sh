#!/usr/bin/env bash

GPU_ID=1
WEIGHTS=../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel

../caffe/build/tools/caffe train \
    -solver ./train/mm_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID

