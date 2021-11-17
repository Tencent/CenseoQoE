// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: infos.cpp
// Email: shaoguowen@tencent.com
// Date: 2021/5/28
// Project: CenseoQoE
//

#include "infos.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/stringbuffer.h>

#include "spdlog/spdlog.h"
#include "utils.h"

int Info::parseModelParamsJson(const std::string &fpath) {
  try {
    assert(CENSEO::file_exist(fpath));
    std::ifstream fin(fpath);
    rapidjson::IStreamWrapper isw(fin);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasParseError() or !d.IsObject()) {
      spdlog::error("Error when parsing json file" + fpath);
      return ERRORCODE;
    }

    // 获取input_process参数
    assert(d.HasMember("input_process"));
    const rapidjson::Value &inputDict = d["input_process"];
    if (inputDict.HasMember("read_mode")) {
      input_param.read_mode = inputDict["read_mode"].GetString();
    } else {
      spdlog::error("read mode must be specified");
      return ERRORCODE;
    }

    if (inputDict.HasMember("read_color")) {
      input_param.read_color = inputDict["read_color"].GetString();
    } else {
      input_param.read_color = "bgr";
    }

    if (inputDict.HasMember("sub_img_dim")) {
      assert(inputDict["sub_img_dim"].IsArray());
      input_param.sub_img_dim =
          std::make_pair(inputDict["sub_img_dim"][0].GetInt(),
                         inputDict["sub_img_dim"][1].GetInt());
    } else {
      spdlog::error("sub_img_dim must be specified");
      return ERRORCODE;
    }
    if (inputDict.HasMember("resize_dim")) {
      assert(inputDict["resize_dim"].IsArray());
      input_param.resize_dim =
          std::make_pair(inputDict["resize_dim"][0].GetInt(),
                         inputDict["resize_dim"][1].GetInt());
    }

    // 获取model参数
    assert(d.HasMember("model"));
    const rapidjson::Value &modelDict = d["model"];
    if (modelDict.HasMember("is_fr")) {
      model_param.is_fr = modelDict["is_fr"].GetBool();
    } else {
      spdlog::error("nr or fr must be specified");
      return ERRORCODE;
    }

    // 获取output_process参数
    assert(d.HasMember("output_process"));
    const rapidjson::Value &outputDict = d["output_process"];
    if (outputDict.HasMember("use_sigmoid")) {
      output_param.use_sigmoid = outputDict["use_sigmoid"].GetBool();
    } else {
      spdlog::info("set use_sigmoid as default value:False");
      output_param.use_sigmoid = false;
    }
    if (outputDict.HasMember("scale")) {
      output_param.scale = outputDict["scale"].GetFloat();
    } else {
      spdlog::info("set scale as default value:1.0");
      output_param.scale = 1.0;
    }
    fin.close();
    return SUCCESSCODE;
  } catch (std::exception &e) {
    spdlog::error("[{}] [{}] parseModelParamsJson error!", __FILE__, __LINE__);
    spdlog::error(e.what());
    return ERRORCODE;
  }
}

void Info::printModelParam() {
  spdlog::info("<<<  Input Process Parameters  >>>");
  spdlog::info("read mode: " + input_param.read_mode);
  spdlog::info("read corlor: " + input_param.read_color);
  spdlog::info("sub_img_dim: [" +
               std::to_string(input_param.sub_img_dim.first) + +"," +
               std::to_string(input_param.sub_img_dim.second) + "]");
  spdlog::info("resize_dim: [" + std::to_string(input_param.resize_dim.first) +
               +"," + std::to_string(input_param.resize_dim.second) + "]");

  spdlog::info("<<<  Model Parameters  >>>");
  spdlog::info("is_fr: " + std::to_string(model_param.is_fr));

  spdlog::info("<<<  Output Process Parameters  >>>");
  spdlog::info("use_sigmoid: " + std::to_string(output_param.use_sigmoid));
  spdlog::info("scale: " + std::to_string(output_param.scale));
}

int Info::parseModelUrlsJson(const std::string &fpath) {
  try {
    assert(CENSEO::file_exist(fpath));
    std::ifstream fin(fpath);
    rapidjson::IStreamWrapper isw(fin);
    rapidjson::Document d;
    d.ParseStream(isw);
    if (d.HasParseError() or !d.IsObject()) {
      spdlog::error("Error when parsing json file" + fpath);
      return ERRORCODE;
    }
    spdlog::debug("<<<  Model URLS  >>>");
    for (auto scene_key = d.MemberBegin(); scene_key != d.MemberEnd();
         scene_key++) {
      std::string scene_name = scene_key->name.GetString();
      for (auto fr_key = scene_key->value.MemberBegin();
           fr_key != scene_key->value.MemberEnd(); fr_key++) {
        std::string fr_name = fr_key->name.GetString();
        for (auto model_key = fr_key->value.MemberBegin();
             model_key != fr_key->value.MemberEnd(); model_key++) {
          std::string model_name = model_key->name.GetString();
          for (auto url_key = model_key->value.MemberBegin();
               url_key != model_key->value.MemberEnd(); url_key++) {
            std::string url_name = url_key->name.GetString();
            std::string url = url_key->value.GetString();
            model_urls[scene_name][fr_name][model_name][url_name] = url;
            spdlog::debug(scene_name + "->" + fr_name + "->" + model_name +
                          "->" + url_name + " : " + url);
          }
        }
      }
    }
    fin.close();
    return SUCCESSCODE;
  } catch (std::exception &e) {
    spdlog::error("[{}] [{}] parseModelUrlsJson error!", __FILE__, __LINE__);
    spdlog::error(e.what());
    return ERRORCODE;
  }
}