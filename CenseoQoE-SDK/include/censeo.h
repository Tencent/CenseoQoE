// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: censeo.h
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//

#ifndef CENSEOQOE_SDK_INCLUDE_CENSEO_H_
#define CENSEOQOE_SDK_INCLUDE_CENSEO_H_

#include <iostream>

#include "infos.h"
#include "libtorch_infer_engine/model_base.h"

class CenseoInference {
 private:
  bool is_fr;
  std::string model_name;
  std::string scene_name;
  std::string fr_name;
  Info *infer_info;
  ModelBase *model;

 public:
  // 默认构造函数，如果用户不指定模型，则使用默认值
  CenseoInference();
  CenseoInference(bool is_fr_, std::string model_name_,
                  std::string scene_name_);
  ~CenseoInference();
  // 初始化
  int init();

  std::unordered_map<std::string, std::vector<float>> predict(
      std::string dst_path, std::string ref_path = "", int subsample = 1);

  std::unordered_map<std::string, std::vector<float>> predict_image_nr(
      std::string &dst_path);

  std::unordered_map<std::string, std::vector<float>> predict_video_nr(
      std::string &dst_path, int subsample);

  std::unordered_map<std::string, std::vector<float>> predict_image_fr(
      std::string &dst_path, std::string &ref_path);

  std::unordered_map<std::string, std::vector<float>> predict_video_fr(
      std::string &dst_path, std::string &ref_path, int subsample);
};

#endif  // CENSEOQOE_SDK_INCLUDE_CENSEO_H_
