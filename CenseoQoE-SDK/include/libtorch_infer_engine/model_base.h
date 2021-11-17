// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: model_base.h
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//

#ifndef CENSEOQOE_SDK_INCLUDE_LIBTORCH_INFER_ENGINE_MODEL_BASE_H_
#define CENSEOQOE_SDK_INCLUDE_LIBTORCH_INFER_ENGINE_MODEL_BASE_H_

#include <unistd.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <exception>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "infos.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

class ModelBase {
 private:
  std::shared_ptr<torch::jit::script::Module> model;
  Info *infer_info;

 public:
  explicit ModelBase(Info *infer_info_);

  cv::Mat transform_color(cv::Mat &image);

  cv::Mat transform_resize_crop(cv::Mat &image);

  torch::Tensor transform_normalize(torch::Tensor &image);

  // 加载模型
  int init_model();

  // 输入预处理
  virtual torch::Tensor input_process(cv::Mat &image);

  // 模型NR预测
  virtual torch::Tensor predict_nr(torch::Tensor &input_dis);

  // 模型FR预测
  virtual torch::Tensor predict_fr(torch::Tensor &input_dis,
                                   torch::Tensor &input_ref);

  // 输出结果后处理，转成std::unordered_map<std::string, std::vector<float> >
  virtual std::unordered_map<std::string, std::vector<float>> output_process(
      torch::Tensor &output);

  virtual std::unordered_map<std::string, std::vector<float>>
  predict_nr_pipeline(std::vector<cv::Mat> &img_dis);

  virtual std::unordered_map<std::string, std::vector<float>>
  predict_fr_pipeline(std::vector<cv::Mat> &img_dis,
                      std::vector<cv::Mat> &img_ref);

  virtual ~ModelBase();
};

#endif  // CENSEOQOE_SDK_INCLUDE_LIBTORCH_INFER_ENGINE_MODEL_BASE_H_
