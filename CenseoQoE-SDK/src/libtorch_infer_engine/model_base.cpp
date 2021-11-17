// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: model_base.cpp
// Email: shaoguowen@tencent.com
// Date: 2021/6/2
// Project: CenseoQoE
//

#include "libtorch_infer_engine/model_base.h"

#include "utils.h"

ModelBase::ModelBase(Info *infer_info_) {
  infer_info = infer_info_;
  init_model();
}

ModelBase::~ModelBase() {
  spdlog::debug("mode base release!");
  model = nullptr;
}

int ModelBase::init_model() {
  std::string model_path = infer_info->model_param.model_path;
  if (model_path.empty()) {
    spdlog::error("please input your model path!");
    return 0;
  }
  bool fret = CENSEO::file_exist(model_path);
  if (!fret) {
    spdlog::error("[{}] [{}] model path:{} not exist!!! please check!!!",
                  __FILE__, __LINE__, model_path);
    return 0;
  } else {
    try {
      spdlog::info("try to load model:{}", model_path);
      model = std::make_shared<torch::jit::script::Module>(
          torch::jit::load(model_path));
      spdlog::info("successfully load model:{}", model_path);
    } catch (std::exception &e) {
      spdlog::info("error load model:{}", model_path);
      spdlog::error("[{}] [{}] err message:{}", __FILE__, __LINE__, e.what());
      return 0;
    }
  }
  return 1;
}

cv::Mat ModelBase::transform_color(cv::Mat &image) {
  cv::Mat img_ret;
  // 如果是gray image， 转为BGR
  if (image.channels() == 1) {
    spdlog::debug("image is gray, transform to BGR mode!");
    cv::cvtColor(image, img_ret, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 4) {
    spdlog::debug("image contain alpha channel, transform to BGR mode!");
    cv::cvtColor(image, img_ret, cv::COLOR_BGRA2BGR);
  } else {
    img_ret = image;
  }
  if (infer_info->input_param.read_color == "rgb") {
    spdlog::debug("transform image from BGR to RGB");
    cv::cvtColor(img_ret, img_ret, cv::COLOR_BGR2RGB);
  }
  return img_ret;
}

cv::Mat ModelBase::transform_resize_crop(cv::Mat &image) {
  cv::Mat img_ret;
  std::string read_mode = infer_info->input_param.read_mode;
  int input_h = infer_info->input_param.sub_img_dim.first;
  int input_w = infer_info->input_param.sub_img_dim.second;
  int h = image.rows, w = image.cols;
  // resize 方式
  if (read_mode == "resize") {
    spdlog::debug("transform mode:resize");
    if ((h - w) * (input_h - input_w) < 0) {
      // 使用逆时针旋转90度
      cv::rotate(image, img_ret, 2);
    } else {
      img_ret = image;
    }
    cv::resize(img_ret, img_ret, cv::Size(input_w, input_h));
    return img_ret;
  } else if (read_mode == "random_crop") {
    spdlog::info("transform mode:random_crop");
    if ((h - input_h) < 0 || (w - input_w) < 0) {
      spdlog::error(
          "[{}] [{}] original image size should be larger than input size!!!"
          " please change another read mode!!!",
          __FILE__, __LINE__);
      return img_ret;
    }
    int h_i = (h - input_h) / 2;
    int w_i = (w - input_w) / 2;
    img_ret = image(cv::Rect(w_i, h_i, input_w, input_h)).clone();
    return img_ret;
  } else if (read_mode == "resize_random_crop") {
    int resize_h = infer_info->input_param.resize_dim.first;
    int resize_w = infer_info->input_param.resize_dim.second;
    spdlog::debug("transform mode:resize_random_crop");
    if ((h - w) * (resize_h - resize_w) < 0) {
      // 使用逆时针旋转90度
      spdlog::debug("rotate image");
      cv::rotate(image, img_ret, 2);
    } else {
      img_ret = image;
    }
    h = img_ret.rows;
    w = img_ret.cols;
    float w_scale = 1.0 * resize_w / w;
    float h_scale = 1.0 * resize_h / h;
    float scale = std::max(h_scale, w_scale);
    int sub_img_h = static_cast<int>(scale * h),
        sub_img_w = static_cast<int>(scale * w);
    cv::resize(img_ret, img_ret, cv::Size(sub_img_w, sub_img_h));
    int h_i = (sub_img_h - input_h) / 2;
    int w_i = (sub_img_w - input_w) / 2;
    img_ret = img_ret(cv::Rect(w_i, h_i, input_w, input_h)).clone();
    return img_ret;
  } else if (read_mode == "resize_padding") {
    spdlog::debug("transform mode:resize_padding");
    if ((h - w) * (input_h - input_w) < 0) {
      // 使用逆时针旋转90度
      cv::rotate(image, img_ret, 2);
    } else {
      img_ret = image;
    }
    h = img_ret.rows;
    w = img_ret.cols;
    float w_scale = 1.0 * input_w / w;
    float h_scale = 1.0 * input_h / h;
    float scale = std::min(h_scale, w_scale);
    int sub_img_h = static_cast<int>(scale * h),
        sub_img_w = static_cast<int>(scale * w);
    cv::resize(img_ret, img_ret, cv::Size(sub_img_w, sub_img_h));
    int top_pad = (input_h - sub_img_h) / 2;
    int bottom_pad = input_h - sub_img_h - top_pad;
    int left_pad = (input_w - sub_img_w) / 2;
    int right_pad = input_w - sub_img_w - left_pad;
    cv::copyMakeBorder(img_ret, img_ret, top_pad, bottom_pad, left_pad,
                       right_pad, cv::BORDER_CONSTANT, cv::Scalar(0));
    return img_ret;
  } else {
    spdlog::warn("such read mode:{} have not been implemented!", read_mode);
    return img_ret;
  }
}

torch::Tensor ModelBase::transform_normalize(torch::Tensor &image) {
  torch::Tensor img_norm = image.div(255);
  return img_norm;
}

torch::Tensor ModelBase::input_process(cv::Mat &image) {
  cv::Mat img_ret = transform_color(image);
  img_ret = transform_resize_crop(img_ret);
  torch::Tensor img_tensor = torch::from_blob(
      img_ret.data, {1, img_ret.rows, img_ret.cols, 3}, torch::kByte);
  img_tensor = img_tensor.permute({0, 3, 1, 2});
  img_tensor = img_tensor.toType(torch::kFloat);
  img_tensor = transform_normalize(img_tensor);
  return img_tensor;
}

torch::Tensor ModelBase::predict_nr(torch::Tensor &inputs_dis) {
  torch::Tensor output;
  if (model != nullptr) {
    try {
      output = model->forward({inputs_dis}).toTensor();
    } catch (std::exception &e) {
      spdlog::error("[{}] [{}] inference error!", __FILE__, __LINE__);
      spdlog::error(e.what());
    }

  } else {
    spdlog::error("[{}] [{}] model is nullptr, please check!", __FILE__,
                  __LINE__);
  }
  return output;
}

torch::Tensor ModelBase::predict_fr(torch::Tensor &inputs_dis,
                                    torch::Tensor &inputs_ref) {
  torch::Tensor output;
  if (model != nullptr) {
    try {
      torch::Tensor inputs = torch::cat({inputs_dis, inputs_ref}, 1);
      output = model->forward({inputs}).toTensor();
    } catch (std::exception &e) {
      spdlog::error("[{}] [{}] inference error!", __FILE__, __LINE__);
      spdlog::error(e.what());
    }

  } else {
    spdlog::error("[{}] [{}] model is nullptr, please check!", __FILE__,
                  __LINE__);
  }
  return output;
}

std::unordered_map<std::string, std::vector<float>> ModelBase::output_process(
    torch::Tensor &output) {
  std::unordered_map<std::string, std::vector<float>> ret;
  // 判断是否使用sigmoid 作为模型最终的输出
  torch::Tensor outputs;
  if (infer_info->output_param.use_sigmoid) {
    outputs = output.sigmoid().view({-1});
  } else {
    outputs = output.view({-1});
  }
  outputs = outputs * infer_info->output_param.scale;  // scale output
  int n = outputs.size(0);
  std::vector<float> tmp(n, 0);
  for (int i = 0; i < n; i++) {
    tmp[i] = outputs[i].item<float>();
  }
  ret["mos"] = tmp;
  return ret;
}

std::unordered_map<std::string, std::vector<float>>
ModelBase::predict_nr_pipeline(std::vector<cv::Mat> &imgs_dis) {
  std::vector<torch::Tensor> imgs_dis_in;
  for (auto img : imgs_dis) {
    imgs_dis_in.push_back(input_process(img));
  }
  torch::Tensor imgs_dis_cat = torch::cat({imgs_dis_in}, 0);
  torch::Tensor output = predict_nr(imgs_dis_cat);
  std::unordered_map<std::string, std::vector<float>> ret =
      output_process(output);
  return ret;
}

std::unordered_map<std::string, std::vector<float>>
ModelBase::predict_fr_pipeline(std::vector<cv::Mat> &imgs_dis,
                               std::vector<cv::Mat> &imgs_ref) {
  std::vector<torch::Tensor> imgs_dis_in;
  for (auto img : imgs_dis) {
    imgs_dis_in.push_back(input_process(img));
  }
  torch::Tensor imgs_dis_cat = torch::cat({imgs_dis_in}, 0);

  std::vector<torch::Tensor> imgs_ref_in;
  for (auto img : imgs_ref) {
    imgs_ref_in.push_back(input_process(img));
  }
  torch::Tensor imgs_ref_cat = torch::cat({imgs_ref_in}, 0);
  torch::Tensor output = predict_fr(imgs_dis_cat, imgs_ref_cat);
  std::unordered_map<std::string, std::vector<float>> ret =
      output_process(output);
  return ret;
}