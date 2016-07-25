#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename Dtype>
void caffe_word_sum(const Blob<Dtype>* in, WordSumParameter* word_sum_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Groups
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < out->channels(); g++){
      for (int k = 0; k < out->height(); k++){
        for (int i = 0; i < in->channels(); i++){
            out_data[out->offset(n, g, k)] +=
                in_data[in->offset(n, i, k)]
                * weight_data[weights[0]->offset(g, i)];

        }
      }
    }
  }  // 
}

template void caffe_word_sum(const Blob<float>* in,
    WordSumParameter* word_sum_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_word_sum(const Blob<double>* in,
    WordSumParameter* word_sum_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class WordSumLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WordSumLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 4, 6, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~WordSumLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WordSumLayerTest, TestDtypesAndDevices);

TYPED_TEST(WordSumLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WordSumParameter* word_sum_param =
      layer_param.mutable_word_sum_param();
  word_sum_param->set_num_output(10);
  shared_ptr<WordSumLayer<Dtype> > layer(
      new WordSumLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

TYPED_TEST(WordSumLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    WordSumParameter* word_sum_param =
        layer_param.mutable_word_sum_param();
    word_sum_param->set_num_output(10);
    word_sum_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<WordSumLayer<Dtype> > layer(
        new WordSumLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count = this->blob_top_->count();
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_word_sum(this->blob_bottom_, word_sum_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(WordSumLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    WordSumParameter* word_sum_param =
        layer_param.mutable_word_sum_param();
    word_sum_param->set_num_output(10);
    word_sum_param->mutable_weight_filler()->set_type("gaussian");
    WordSumLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
