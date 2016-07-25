#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MatrixProdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data0 = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data1 = bottom[1]->gpu_data();
  for (int i=0; i<bottom[1]->shape(0); ++i){
    // B0 (K*M) * B1 (M*N)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
        bottom_data0+i*K_*M_, bottom_data1+i*M_*N_, (Dtype)0., top_data+i*K_*N_);
  }
}

template <typename Dtype>
void MatrixProdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    for (int i=0; i<bottom[1]->shape(0); ++i){
      // B0^T (M*K) * D (K*N)
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          bottom_data+i*K_*M_, top_diff+i*K_*N_, (Dtype)0.,
          bottom[1]->mutable_gpu_diff()+i*N_*M_);
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[1]->gpu_data();
    // Gradient with respect to bottom data
    for (int i=0; i<bottom[1]->shape(0); ++i){
      // D (K*N) * B1^T (N*M)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, M_, N_, (Dtype)1.,
          top_diff+i*N_*K_, bottom_data+i*N_*M_,  (Dtype)0.,
          bottom[0]->mutable_gpu_diff()+i*K_*M_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MatrixProdLayer);

}  // namespace caffe
