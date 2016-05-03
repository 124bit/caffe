import sys, os
sys.path.append(os.path.join(os.getcwd(), 'python'))
import caffe
# caffe.prepare_python_threads()
net = caffe.Net('python_gil_test/net.prototxt', caffe.TRAIN)
net.forward()
net.backward()

solver = caffe.SGDSolver('python_gil_test/custom_lenet_solver.prototxt')
solver.solve()
print 'OK'