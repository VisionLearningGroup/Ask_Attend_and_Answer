#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void MatrixProdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size()==2) << "Bottom should be 2";
}

template <typename Dtype>
void MatrixProdLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[1]->shape(1);
  N_ = bottom[1]->shape(2);
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(1));
  if (bottom[0]->shape(0)==1){
    // For 1*L*M * L*M*N = 1*L*N
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(0));
    K_ = bottom[0]->shape(0); 
  } else {
    // For L*K*M * L*M*N = L*K*N
    CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
    K_ = bottom[0]->shape(1); 
  }
  vector<int> top_shape = bottom[0]->shape();
  top_shape[2] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MatrixProdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i=0; i<bottom[1]->shape(0); ++i){
    // B0 (K*M) * B1 (M*N)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
        bottom_data0+i*K_*M_, bottom_data1+i*M_*N_, (Dtype)0., top_data+i*K_*N_);
  }
}

// Backward still ongoing
template <typename Dtype>
void MatrixProdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    for (int i=0; i<bottom[1]->shape(0); ++i){
      // B0^T (M*K) * D (K*N)
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          bottom_data+i*K_*M_, top_diff+i*K_*N_, (Dtype)0., 
          bottom[1]->mutable_cpu_diff()+i*N_*M_);
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[1]->cpu_data();
    // Gradient with respect to bottom data
    for (int i=0; i<bottom[1]->shape(0); ++i){
      // D (K*N) * B1^T (N*M)
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, M_, N_, (Dtype)1.,
          top_diff+i*N_*K_, bottom_data+i*N_*M_,  (Dtype)0.,
          bottom[0]->mutable_cpu_diff()+i*K_*M_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MatrixProdLayer);
#endif

INSTANTIATE_CLASS(MatrixProdLayer);
REGISTER_LAYER_CLASS(MatrixProd);

}  // namespace caffe
