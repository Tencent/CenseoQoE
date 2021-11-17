// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: options.cpp
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//

#include "options.h"

#include <algorithm>
#include <string>

#include "spdlog/spdlog.h"
#include "utils.h"

int Options::parseOptions(int _argc, const char *const *_argv) {
  for (int i = 1; i < _argc; i++) {
    std::string option(_argv[i]);
    std::string optionValue;
    if (i < _argc - 1 && option.rfind("--", 0) == 0) {
      optionValue = option.substr(2);
      if (option == "--dis_file") {
        dis_file = std::string(_argv[i + 1]);
      }
      if (option == "--ref_file") {
        ref_file = std::string(_argv[i + 1]);
      }
      if (option == "--model_name") {
        model_name = std::string(_argv[i + 1]);
      }
      if (option == "--model_scene") {
        model_scene = std::string(_argv[i + 1]);
      }
      if (option == "--subsample") {
        subsample = std::stoi(_argv[i + 1]);
      }
      if (option == "--result_dir") {
        result_dir = std::string(_argv[i + 1]);
      }
      if (option == "--save_name") {
        save_name = std::string(_argv[i + 1]);
      }
    }
    if (option == "--fr") {
      is_fr = true;
    }
  }
  // 检查传入的参数是否合法
  if (std::find(MODEL_NAMES.begin(), MODEL_NAMES.end(), model_name) ==
      MODEL_NAMES.end()) {
    spdlog::error("[{}] [{}] model_name is not ALLOWED: " + model_name,
                  __FILE__, __LINE__);
    return 1;
  }
  if (std::find(MODEL_SCENES.begin(), MODEL_SCENES.end(), model_scene) ==
      MODEL_SCENES.end()) {
    spdlog::error("[{}] [{}] model_scene is not ALLOWED: " + model_scene,
                  __FILE__, __LINE__);
    return 1;
  }
  if (!CENSEO::file_exist(dis_file)) {
    spdlog::error("[{}] [{}] please make sure file exists: " + dis_file,
                  __FILE__, __LINE__);
    return 1;
  }
  if (is_fr && !CENSEO::file_exist(ref_file)) {
    spdlog::error(
        "[{}] [{}] please make sure file exists in FR mode: " + ref_file,
        __FILE__, __LINE__);
    return 1;
  }
  if (result_dir.empty()) {
    result_dir = "results";
  }
  spdlog::info("make results dir: " + result_dir);
  CENSEO::pmkdir(result_dir.c_str());
  if (save_name.empty()) {
    // 如果不指定save_name, 则生成一个
    std::string dis_name = dis_file.substr(dis_file.find_last_of('/') + 1);
    save_name = dis_name + "-" + model_name + "-" + model_scene + "-" +
                std::to_string(is_fr) + "-" + std::to_string(subsample) +
                ".json";
  }
  if (subsample < 1) {
    spdlog::error(
        "[{}] [{}] please make sure Subsample greater than 0: " + ref_file,
        __FILE__, __LINE__);
    return 1;
  }
  return 0;
}
