#include <vector>
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

// JFC: Modified to implement a circular buffer filled by an external thread

#ifdef __BIDMACH__
#ifdef _MSC_VER
#include <windows.h>
#else
#include <unistd.h>
#define Sleep(x) usleep((x)*1000)
#endif
#else
#endif

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  this->datum_channels_ = this->layer_param_.memory_data_param().channels();
  this->datum_height_ = this->layer_param_.memory_data_param().height();
  this->datum_width_ = this->layer_param_.memory_data_param().width();
  this->datum_size_ = this->datum_channels_ * this->datum_height_ *
      this->datum_width_;
  CHECK_GT(batch_size_ * this->datum_size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  (*top)[0]->Reshape(batch_size_, this->datum_channels_, this->datum_height_,
                     this->datum_width_);
  (*top)[1]->Reshape(batch_size_, 1, 1, 1);
  n_ = this->layer_param_.memory_data_param().lookahead() * batch_size_;
  added_data_.Reshape(n_, this->datum_channels_, this->datum_height_,
                      this->datum_width_);
  added_label_.Reshape(n_, 1, 1, 1);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
#ifdef __BIDMACH__
  Reset(added_data_.mutable_cpu_data(), added_label_.mutable_cpu_data(), n_);
#endif
}

#ifndef __BIDMACH__
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add Datum when earlier ones haven't been consumed"
      << " by the upper layers";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  CHECK_LE(num, batch_size_) <<
      "The number of added datum must be no greater than the batch size";

  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int batch_item_id = 0; batch_item_id < num; ++batch_item_id) {
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(
        batch_item_id, datum_vector[batch_item_id], this->mean_, top_data);
    top_label[batch_item_id] = datum_vector[batch_item_id].label();
  }
  // num_images == batch_size_
  Reset(top_data, top_label, batch_size_);
  has_new_data_ = true;
}

#else
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  while (num >= (pos_ - write_pos_ + n_) % n_) Sleep(1);   // Not enough space to add the item, so wait

  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int batch_item_id = 0; batch_item_id < num; ++batch_item_id) {
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(
        write_pos_, datum_vector[batch_item_id], this->mean_, top_data);
    top_label[write_pos_] = datum_vector[batch_item_id].label();
    write_pos_ = (write_pos_ + 1) % n_;
  }
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddData(Dtype *A, int num, int nchannels, int width, int height) {
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add";
  CHECK_EQ(nchannels, this->datum_channels_)<< "MemoryDataLayer: wrong number of channels";
  CHECK_EQ(width, this->datum_width_)<< "MemoryDataLayer: wrong width";
  CHECK_EQ(height, this->datum_height_)<< "MemoryDataLayer: wrong height";
  while (num >= (pos_ - write_pos_ + n_) % n_) Sleep(1);   // Not enough space to add the item, so wait

  Dtype* top_data = added_data_.mutable_cpu_data();
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int batch_item_id = 0; batch_item_id < num; ++batch_item_id) {
    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(write_pos_, A+(batch_item_id * this->datum_size_), this->mean_, top_data);
    top_label[write_pos_] = datum_vector[batch_item_id].label();
    write_pos_ = (write_pos_ + 1) % n_;
  }
}
#endif

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
#ifdef __BIDMACH__
  write_pos_ = 0;
#endif
}

#ifndef __BIDMACH__
template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  (*top)[0]->set_cpu_data(data_ + pos_ * this->datum_size_);
  (*top)[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
  has_new_data_ = false;
}
#else
template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  while ((write_pos_ - pos_ + n_) % n_ < batch_size_) {  // Wait for enough data to read
    Sleep(1);
  }
  CHECK(data_) << "MemoryDataLayer needs to be initalized";
  memcpy((*top)[0]->mutable_cpu_data(), data_ + pos_ * this->datum_size_, batch_size_ * this->datum_size_ * sizeof(Dtype));
  memcpy((*top)[1]->mutable_cpu_data(), labels_ + pos_, batch_size_ * sizeof(Dtype));
  pos_ = (pos_ + batch_size_) % n_;
}
#endif

INSTANTIATE_CLASS(MemoryDataLayer);

}  // namespace caffe
