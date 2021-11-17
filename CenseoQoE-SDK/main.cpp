// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: main.cpp
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//
#include <iostream>
#include <unordered_map>

#include "censeo.h"
#include "options.h"
#include "utils.h"

int main(int _argc, const char *const *_argv) {
  spdlog::set_level(spdlog::level::info);
  // 传入参数
  Options opts;
  opts.parseOptions(_argc, _argv);
  // 模型预测
  auto *c_infer =
      new CenseoInference(opts.is_fr, opts.model_name, opts.model_scene);
  std::unordered_map<std::string, std::vector<float>> ret =
      c_infer->predict(opts.dis_file, opts.ref_file, opts.subsample);
  // 保存结果
  std::string save_path = opts.result_dir + "/" + opts.save_name;
  CENSEO::save_result(ret, save_path);
  CENSEO::print_result(ret);
  delete c_infer;
  return 0;
}
