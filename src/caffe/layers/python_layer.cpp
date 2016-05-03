#ifdef WITH_PYTHON_LAYER

#include "caffe/layers/python_layer.hpp"

#include <boost/thread.hpp>
#include <vector>

namespace caffe {



void prepare_python_threads() {
  if (!PyEval_ThreadsInitialized()) {
        PyEval_InitThreads();
        PyThreadState* mainPyThread = PyEval_SaveThread();
    }
}



template <typename Dtype>
void PythonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  self_.attr("param_str") = bp::str(
      this->layer_param_.python_param().param_str());
  self_.attr("setup")(bottom, top);
  PyGILState_Release(gstate);
}

template <typename Dtype>
void PythonLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  self_.attr("reshape")(bottom, top);
  PyGILState_Release(gstate);
}

template <typename Dtype>
void PythonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  self_.attr("forward")(bottom, top);
  PyGILState_Release(gstate);
}

template <typename Dtype>
void PythonLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  self_.attr("backward")(top, propagate_down, bottom);
  PyGILState_Release(gstate);
}

INSTANTIATE_CLASS(PythonLayer);

}  // namespace caffe

#endif  // #ifdef WITH_PYTHON_LAYER