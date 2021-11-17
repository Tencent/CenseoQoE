// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: utils.h
// Email: shaoguowen@tencent.com
// Date: 2021/5/27
// Project: CenseoQoE
//

#ifndef CENSEOQOE_SDK_INCLUDE_UTILS_H_
#define CENSEOQOE_SDK_INCLUDE_UTILS_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace CENSEO {
int is_video(const std::string &fpath);

int is_image(const std::string &fpath);

std::string to_lower(std::string s);

std::string to_upper(std::string s);

void update_map(std::unordered_map<std::string, std::vector<float>> &dst,
                std::unordered_map<std::string, std::vector<float>> src);

int save_result(std::unordered_map<std::string, std::vector<float>> &outputs,
                std::string &save_path);

void make_dirs(const std::string &dir_path);

void print_result(std::unordered_map<std::string, std::vector<float>> &ret);

std::vector<cv::Mat> read_video(std::string &vpath);

bool file_exist(const std::string &fpath);

std::pair<bool, std::string> download_file(const std::string &url,
                                           const std::string &save_dir);

std::pair<bool, std::string> unzip_file(const std::string &zip_path,
                                        const std::string &extract_dir);

int pmkdir(const char *path);

std::string get_cur_abs_dir();
}  // namespace CENSEO

#endif  // CENSEOQOE_SDK_INCLUDE_UTILS_H_
