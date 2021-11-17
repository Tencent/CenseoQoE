// Copyright 2021 Tencent Inc.  All rights reserved.
//
// ref. from TNN:https://github.com/Tencent/TNN
//

#include "timer.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;

void Timer::Start() { start_ = system_clock::now(); }

void Timer::Stop() { stop_ = system_clock::now(); }

double Timer::GetTime() {
  return duration_cast<microseconds>(stop_ - start_).count() / 1000.0f;
}

void Timer::Reset() { stop_ = start_ = system_clock::now(); }
