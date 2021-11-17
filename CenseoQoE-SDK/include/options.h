// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: options.h
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//

#ifndef CENSEOQOE_SDK_INCLUDE_OPTIONS_H_
#define CENSEOQOE_SDK_INCLUDE_OPTIONS_H_

#include <iostream>
#include <vector>

class Options {
 private:
  const std::vector<std::string> MODEL_SCENES = {"PGC", "UGC", "GAMES"};
  const std::vector<std::string> MODEL_NAMES = {"general", "compressed"};

 public:
  bool is_fr = false;  // 是否有参
  int subsample = 1;  // 如果视频预测抽帧频率，1表示每一帧都预测
  std::string dis_file;  // 要预测的失真文件路径
  std::string ref_file;  // 失真文件对应的参考文件（有参情况下提供）
  std::string model_scene;   // 模型类型（UGC/PGC/GAMES等)
  std::string model_name;  // 模型场景（compressed/general等)
  std::string result_dir;   // 结果保存路径
  std::string save_name;    // 保存的json名称
 public:
  int parseOptions(int _argc, const char *const *_argv);
};

#endif  // CENSEOQOE_SDK_INCLUDE_OPTIONS_H_
