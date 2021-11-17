// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: censeo.cpp
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//

#include "censeo.h"

#include <torch/script.h>

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include "infos.h"
#include "spdlog/spdlog.h"
#include "timer.h"
#include "utils.h"

CenseoInference::CenseoInference() {
  is_fr = false;
  scene_name = "UGC";
  model_name = "general";
  fr_name = is_fr ? "fr" : "nr";
  infer_info = new Info();
  init();
}

CenseoInference::CenseoInference(bool is_fr_, std::string model_name_,
                                 std::string scene_name_) {
  is_fr = is_fr_;
  model_name = model_name_;
  scene_name = scene_name_;
  fr_name = is_fr ? "fr" : "nr";
  infer_info = new Info();
  init();
}

CenseoInference::~CenseoInference() {
  if (infer_info != nullptr) {
    delete infer_info;
    infer_info = nullptr;
  }
  if (model != nullptr) {
    delete model;
    model = nullptr;
  }
}

int CenseoInference::init() {
  // 先获取model path
  std::string model_urls_path =
      CENSEO::get_cur_abs_dir() + "/../assets/model_urls.json";
  assert(CENSEO::file_exist(
      model_urls_path));  // 确保该json文件存在，这里写死了在assets/model_urls.json
  infer_info->parseModelUrlsJson(model_urls_path);
  std::string model_url =
      infer_info->model_urls[scene_name][fr_name][model_name]["model_url"];
  std::string config_url =
      infer_info->model_urls[scene_name][fr_name][model_name]["config_url"];
  // 根据model path下载模型文件并解压, 所有的模型默认保存在checkpoints下
  std::string checkpoint_dir = CENSEO::get_cur_abs_dir() + "/../checkpoints";
  checkpoint_dir += "/" + scene_name + "/" + fr_name + "/" + model_name;
  // 下载模型
  std::pair<bool, std::string> ret_dload_model =
      CENSEO::download_file(model_url, checkpoint_dir);
  if (!ret_dload_model.first) {
    spdlog::error("[{}] [{}] failed to download model", __FILE__, __LINE__);
    return 1;
  }

  std::pair<bool, std::string> ret_dload_config =
      CENSEO::download_file(config_url, checkpoint_dir);
  if (!ret_dload_config.first) {
    spdlog::error("[{}] [{}] failed to download config", __FILE__, __LINE__);
    return 1;
  }

  // 解析config.json
  infer_info->parseModelParamsJson(checkpoint_dir + "/" + "config.json");
  infer_info->model_param.model_path = checkpoint_dir + "/" + "model.pt";
  // 初始化模型
  spdlog::info("<<<  Start Loading Model  >>>");
  torch::set_num_threads(8);
  model = new ModelBase(
      infer_info);  // 这里预留多态实现，如果新添加模型，继承ModelBase实现
  spdlog::info("<<<  Successfully Loading Model  >>>");
  return 0;
}

std::unordered_map<std::string, std::vector<float>> CenseoInference::predict(
    std::string dst_path, std::string ref_path, int subsample) {
  spdlog::info("start predicting >>>");
  std::unordered_map<std::string, std::vector<float>> ret;
  if (!is_fr && CENSEO::is_video(dst_path)) {
    return predict_video_nr(dst_path, subsample);
  } else if (!is_fr && CENSEO::is_image(dst_path)) {
    return predict_image_nr(dst_path);
  } else if (is_fr && CENSEO::is_video(dst_path) &&
             CENSEO::is_video(ref_path)) {
    return predict_video_fr(dst_path, ref_path, subsample);
  } else if (is_fr && CENSEO::is_image(dst_path) &&
             CENSEO::is_image(ref_path)) {
    return predict_image_fr(dst_path, ref_path);
  } else {
    spdlog::error("[{}] [{}] such file {} not supported!", __FILE__, __LINE__,
                  dst_path);
  }
  return ret;
}

std::unordered_map<std::string, std::vector<float>>
CenseoInference::predict_image_nr(std::string &dst_path) {
  Timer tictok;
  tictok.Start();
  spdlog::info("reading image:{}", dst_path);
  std::vector<cv::Mat> imgs_dis;
  cv::Mat image = cv::imread(dst_path);
  imgs_dis.push_back(image);
  std::unordered_map<std::string, std::vector<float>> ret =
      model->predict_nr_pipeline(imgs_dis);
  tictok.Stop();
  spdlog::info("image path:{}, inference time:{} ms ", dst_path,
               tictok.GetTime());
  return ret;
}

std::unordered_map<std::string, std::vector<float>>
CenseoInference::predict_video_nr(std::string &dst_path, int subsample) {
  std::unordered_map<std::string, std::vector<float>> ret;
  spdlog::info("reading video:{}", dst_path);
  cv::VideoCapture vc(dst_path);
  if (!vc.isOpened()) {
    spdlog::error("[{}] [{}] cannot open video:{}", __FILE__, __LINE__,
                  dst_path);
    return ret;
  }
  std::vector<cv::Mat> images;
  int batch_size = 8;
  int i = 0, j = 0;
  while (true) {
    if (images.size() == batch_size) {
      spdlog::debug("clear images vector!");
      images.clear();
    }
    cv::Mat frame;
    vc >> frame;
    if (frame.empty()) {
      spdlog::info("done reading video:{}", dst_path);
      break;
    }
    if (j % subsample == 0) {
      images.push_back(frame);
    }
    j++;
    if (images.size() == batch_size) {
      Timer tictok;
      tictok.Start();
      auto outputs = model->predict_nr_pipeline(images);
      i += 1;
      CENSEO::update_map(ret, outputs);
      tictok.Stop();
      spdlog::info("batch:{}, batch size:{}, inference time:{} ms ", i,
                   images.size(), tictok.GetTime());
    }
  }
  if (!images.empty()) {
    spdlog::debug("batch {}: predicting>>>", i);
    auto outputs = model->predict_nr_pipeline(images);
    CENSEO::update_map(ret, outputs);
  }
  vc.release();
  return ret;
}

std::unordered_map<std::string, std::vector<float>>
CenseoInference::predict_image_fr(std::string &dst_path,
                                  std::string &ref_path) {
  Timer tictok;
  tictok.Start();
  spdlog::info("reading dst image:{}", dst_path);
  std::vector<cv::Mat> imgs_dis;
  cv::Mat img_dst = cv::imread(dst_path);
  imgs_dis.push_back(img_dst);
  spdlog::info("reading ref image:{}", ref_path);
  std::vector<cv::Mat> imgs_ref;
  cv::Mat img_ref = cv::imread(ref_path);
  imgs_ref.push_back(img_ref);
  std::unordered_map<std::string, std::vector<float>> ret =
      model->predict_fr_pipeline(imgs_dis, imgs_ref);
  tictok.Stop();
  spdlog::info("image path:{}, inference time:{} ms ", dst_path,
               tictok.GetTime());
  return ret;
}

std::unordered_map<std::string, std::vector<float>>
CenseoInference::predict_video_fr(std::string &dst_path, std::string &ref_path,
                                  int subsample) {
  std::unordered_map<std::string, std::vector<float>> ret;
  spdlog::info("start reading dst video:{}", dst_path);
  cv::VideoCapture vc1(dst_path);
  if (!vc1.isOpened()) {
    spdlog::error("[{}] [{}] cannot open video:{}", __FILE__, __LINE__,
                  dst_path);
    return ret;
  }
  cv::VideoCapture vc2(ref_path);
  spdlog::info("start reading ref video:{}", ref_path);
  if (!vc2.isOpened()) {
    spdlog::error("[{}] [{}] cannot open video:{}", __FILE__, __LINE__,
                  ref_path);
    return ret;
  }
  std::vector<cv::Mat> images_dst, images_ref;
  int batch_size = 8;
  int i = 0, j = 0;
  while (true) {
    if (images_dst.size() == batch_size) {
      spdlog::debug("clear images vector!");
      images_dst.clear();
      images_ref.clear();
    }
    cv::Mat frame_dst, frame_ref;
    vc1 >> frame_dst;
    vc2 >> frame_ref;
    if (frame_dst.empty() || frame_ref.empty()) {
      spdlog::info("done reading video:{}, {}", dst_path, ref_path);
      break;
    }
    if (j % subsample == 0) {
      images_dst.push_back(frame_dst);
      images_ref.push_back(frame_ref);
    }
    j++;
    if (images_dst.size() == batch_size) {
      Timer tictokFr;
      tictokFr.Start();
      i += 1;
      auto outputs_fr = model->predict_fr_pipeline(images_dst, images_ref);
      CENSEO::update_map(ret, outputs_fr);
      tictokFr.Stop();
      spdlog::info("batch:{}, batch size:{}, inference time:{} ms ", i,
                   images_dst.size(), tictokFr.GetTime());
    }
  }
  if (!images_dst.empty()) {
    spdlog::info("batch {}: predicting>>>", i);
    auto outputs = model->predict_fr_pipeline(images_dst, images_ref);
    CENSEO::update_map(ret, outputs);
  }
  vc1.release();
  vc2.release();
  return ret;
}