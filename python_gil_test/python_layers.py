import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))

import caffe
import numpy as np


class SimplePythonLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need one input.")

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels)  # , bottom[0].height, bottom[0].width)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = top[0].diff


class EuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


class PythonDummyData(caffe.Layer):
    def setup(self, bottom, top):
        assert(len(top)==1) # one output

    def reshape(self, bottom, top):
        top[0].reshape(10,3,2)

    def forward(self, bottom, top):
        top[0].data[...] = np.random.randn(10, 3, 2)

    def backward(self,top, propagate_down, bottom):
        pass