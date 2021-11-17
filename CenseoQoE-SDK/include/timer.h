// Copyright 2021 Tencent Inc.  All rights reserved.
//
// ref from TNN: https://github.com/Tencent/TNN
//

#ifndef CENSEOQOE_SDK_INCLUDE_TIMER_H_
#define CENSEOQOE_SDK_INCLUDE_TIMER_H_

#include <chrono>

using std::chrono::system_clock;
using std::chrono::time_point;

class Timer {
 public:
  Timer() {}

  void Start();

  void Stop();

  void Reset();

  double GetTime();

 private:
  time_point<system_clock> start_;
  time_point<system_clock> stop_;
};

#endif  // CENSEOQOE_SDK_INCLUDE_TIMER_H_
