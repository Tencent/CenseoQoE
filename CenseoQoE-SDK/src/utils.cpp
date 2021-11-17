// Copyright 2021 Tencent Inc.  All rights reserved.
//
// Author: shaoguowen
// File: utils.cpp
// Email: shaoguowen@tencent.com
// Date: 2021/5/28
// Project: CenseoQoE
//

#include "utils.h"

#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>

#include "spdlog/spdlog.h"
#include "zip.h"

std::string CENSEO::to_lower(std::string s) {
  std::string ret = s;
  for (int i = 0; i < s.size(); ++i) {
    ret[i] = std::tolower(s[i]);
  }
  return ret;
}

std::string CENSEO::to_upper(std::string s) {
  std::string ret = s;
  for (int i = 0; i < s.size(); ++i) {
    ret[i] = std::toupper(s[i]);
  }
  return ret;
}

int CENSEO::is_video(const std::string &fpath) {
  std::vector<std::string> vnames = {"mp4", "h264", "avi", "mkv", "mov"};
  std::string fext = fpath.substr(fpath.find_last_of('.') + 1);
  fext = to_lower(fext);
  auto ret = std::find(vnames.begin(), vnames.end(), fext);
  if (ret != vnames.end()) {
    return 1;
  } else {
    return 0;
  }
}

int CENSEO::is_image(const std::string &fpath) {
  std::vector<std::string> inames = {"png", "jpg", "jpeg", "bmp", "gif"};
  std::string fext = fpath.substr(fpath.find_last_of('.') + 1);
  fext = to_lower(fext);
  auto ret = std::find(inames.begin(), inames.end(), fext);
  if (ret != inames.end()) {
    return 1;
  } else {
    return 0;
  }
}

void CENSEO::update_map(
    std::unordered_map<std::string, std::vector<float>> &dst,
    std::unordered_map<std::string, std::vector<float>> src) {
  for (auto data : src) {
    auto p = dst.find(data.first);
    if (p == dst.end()) {
      dst[data.first] = data.second;
    } else {
      for (auto i : data.second) {
        dst[data.first].push_back(i);
      }
    }
  }
}

int CENSEO::save_result(
    std::unordered_map<std::string, std::vector<float>> &outputs,
    std::string &save_path) {
  rapidjson::Document doc;
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();
  rapidjson::Value root(rapidjson::kObjectType);
  for (auto r : outputs) {
    rapidjson::Value data(
        rapidjson::kArrayType);  // kArrayType 枚举成员 表示数组
    for (auto i : r.second) {
      rapidjson::Value float_object(rapidjson::kObjectType);
      float_object.SetFloat(i);
      data.PushBack(float_object, allocator);
    }
    std::string key = r.first;
    rapidjson::Value keyname(key.c_str(), doc.GetAllocator());
    root.AddMember(keyname, data, allocator);
  }
  std::ofstream ofs{save_path};
  if (!ofs.is_open()) {
    spdlog::error("[{}] [{}} Could not open file for writing:{}!", __FILE__,
                  __LINE__, save_path);
    return 0;
  }
  rapidjson::OStreamWrapper osw{ofs};
  rapidjson::Writer<rapidjson::OStreamWrapper> writer{osw};
  root.Accept(writer);
  spdlog::info("save results at:{}", save_path);
  return 1;
}

void CENSEO::make_dirs(const std::string &dir_path) {
  int fret = access(dir_path.c_str(), W_OK);
  if (fret == 0) {
    spdlog::info("{} have existed!", dir_path);
  } else {
    spdlog::info("make dir:{}!", dir_path);
    mkdir(dir_path.c_str(), S_IRWXU);
  }
}

void CENSEO::print_result(
    std::unordered_map<std::string, std::vector<float>> &ret) {
  float mos_mean = 0;
  std::vector<float> tmp = ret["mos"];
  if (tmp.empty()) {
    spdlog::info("The result is empty");
    return;
  }
  for (auto i : tmp) {
    mos_mean += i;
  }
  spdlog::info("MOS: {}", mos_mean / tmp.size());
}

std::vector<cv::Mat> CENSEO::read_video(std::string &vpath) {
  std::vector<cv::Mat> ret;
  cv::VideoCapture vc(vpath);
  if (!vc.isOpened()) {
    spdlog::error("[{}] [{}] cannot open video:{}", __FILE__, __LINE__, vpath);
    return ret;
  }
  while (true) {
    cv::Mat frame;
    vc >> frame;
    if (frame.empty()) {
      spdlog::info("done reading video:{}", vpath);
      break;
    }
    spdlog::debug("image shape:{}x{}", frame.rows, frame.cols);
    ret.push_back(frame);
  }
  return ret;
}

bool CENSEO::file_exist(const std::string &fpath) {
  std::ifstream infile(fpath.c_str());
  return infile.good();
}

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t written = fwrite(ptr, size, nmemb, stream);
  return written;
}

int progress_func(char *progress_data, double t, double d, double ultotal,
                  double ulnow) {
  char bar[101] = {0};
  int perc = static_cast<int >(d * 100.0 / t);
  if (perc >= 0 && perc <= 100) {
    for (int i = 0; i < perc; ++i) {
      bar[i] = '>';
    }
    printf("[%-100s] [%3d%%] \r", bar, perc);
    fflush(stdout);
    usleep(50);
  }
  return 0;
}

std::pair<bool, std::string> CENSEO::download_file(
    const std::string &url, const std::string &save_dir) {
  spdlog::info("start downloading url: " + url);
  CURL *curl;
  FILE *fp;
  CURLcode res = CURLE_RECV_ERROR;
  std::string filename = url.substr(url.find_last_of('/') + 1);
  spdlog::info("make save dirs:" + save_dir);
  pmkdir(save_dir.c_str());
  std::string save_path = save_dir + "/" + filename;
  if (file_exist(save_path)) {
    spdlog::info("file exist, do not download again: " + save_path);
    return std::make_pair(true, save_path);
  }
  std::string pregress_data = "Downloading Model:";
  curl = curl_easy_init();
  if (curl) {
    fp = fopen(save_path.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_func);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, pregress_data.c_str());
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);
  }
  if (res == CURLE_OK) {
    spdlog::info("successfully downloading at: " + save_path);
    return std::make_pair(true, save_path);
  } else {
    spdlog::error("[{}] [{}] error downloading: " + url, __FILE__, __LINE__);
    return std::make_pair(false, "");
  }
}

int CENSEO::pmkdir(const char *path) {
  char name[PATH_MAX];
  strcpy(name, path);
  int i, len = strlen(name);

  if (name[len - 1] != '/') {
    strcat(name, "/");
  }

  len = strlen(name);

  for (i = 1; i < len; i++) {
    if (name[i] == '/') {
      name[i] = 0;
      if (access(name, 0) != 0) {
        if (mkdir(name, 0755) == -1) {
          spdlog::error("[{}] [{}] mkdir error", __FILE__, __LINE__);
          return -1;
        }
      }
      name[i] = '/';
    }
  }

  return 0;
}

std::pair<bool, std::string> CENSEO::unzip_file(
    const std::string &zip_path, const std::string &extract_dir) {
  // ref from zeerd's blog: https://blog.zeerd.com/linux-libzip/
  int err = 0;
  char strerr[1024];
  struct zip *z = nullptr;
  std::string extract_path_ret;
  spdlog::info("make extract dirs:" + extract_dir);
  pmkdir(extract_dir.c_str());
  z = zip_open(zip_path.c_str(), ZIP_CREATE, &err);
  if (z != nullptr) {
    zip_int64_t i, c = zip_get_num_entries(z, ZIP_FL_UNCHANGED);
    for (i = 0; i < c; i++) {
      const char *name = zip_get_name(z, i, ZIP_FL_ENC_GUESS);
      std::string extract_path = extract_dir + "/" + name;
      if (extract_path_ret.empty()) {
        // 取第一个作为作为解压的目录
        extract_path_ret = extract_path;
        if (CENSEO::file_exist(extract_path_ret)) {
          // 如果已经解压过了，则不用再重新解压
          return std::make_pair(true, extract_path_ret);
        }
      }
      const char *dtmp = extract_path.c_str();
      char *d = strdup(dtmp);
      if (d != nullptr) {
        char *p = strrchr(d, '/');
        if (p != nullptr) {
          *p = '\0';
          pmkdir(d);
        }
        free(d);
        FILE *fp = fopen(dtmp, "w+b");
        struct zip_file *f = zip_fopen(z, name, 0);
        if (f != nullptr && fp != nullptr) {
          zip_int64_t j, n = 0;
          char buf[8192] = "";
          while ((n = zip_fread(f, buf, sizeof(buf))) > 0) {
            for (j = 0; j < n; j++) {
              putc(buf[j], fp);
            }
          }
          fclose(fp);
          zip_fclose(f);
        }
      } else {
        spdlog::error("[{}] [{}] memory low", __FILE__, __LINE__);
      }
    }

    err = zip_close(z);
  } else {
    zip_error_to_str(strerr, 1024, err, errno);
    spdlog::error("[{}] [{}] operated zip fail for {}", __FILE__, __LINE__,
                  strerr);
  }
  return std::make_pair(err == 0, extract_path_ret);
}

std::string CENSEO::get_cur_abs_dir() {
  std::string cur_file_path = __FILE__;
  return cur_file_path.substr(0, cur_file_path.find_last_of('/'));
}