// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: infos.h
// Email: shaoguowen@tencent.com
// Date: 2021/5/28
// Project: CenseoQoE
//

#ifndef CENSEOQOE_SDK_INCLUDE_INFOS_H_
#define CENSEOQOE_SDK_INCLUDE_INFOS_H_

#include <iostream>
#include <unordered_map>

// 模型运行的参数设置，后续要添加额外的参数在这里添加
struct InputProcessParam {
  std::string read_mode;
  std::string read_color;
  std::pair<int, int> sub_img_dim;
  std::pair<int, int> resize_dim;
};

struct ModelParam {
  std::string model_path;
  bool is_fr;
};

struct OutputProcessParam {
  bool use_sigmoid;
  float scale;
};

class Info {
 private:
  const int ERRORCODE = 1;
  const int SUCCESSCODE = 0;

 public:
  InputProcessParam input_param;
  ModelParam model_param;
  OutputProcessParam output_param;
  std::unordered_map<
      std::string,
      std::unordered_map<
          std::string,
          std::unordered_map<std::string,
                             std::unordered_map<std::string, std::string>>>>
      model_urls;

 public:
  int parseModelParamsJson(const std::string &fpath);
  void printModelParam();
  int parseModelUrlsJson(const std::string &fpath);
};

#endif  // CENSEOQOE_SDK_INCLUDE_INFOS_H_
