/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "squeeze_test.h"
#include "space_to_batch_nd_test.h"
#include "transpose_test.h"
#include "one_hot_test.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
}  // namespace

void setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
}

void loop() {
  TF_LITE_REPORT_ERROR(error_reporter, "INIT_TESTING");

  squeeze_test(error_reporter);
  space_to_batch_nd_test(error_reporter);
  transpose_test(error_reporter);
  one_hot_test(error_reporter);

  TF_LITE_REPORT_ERROR(error_reporter, "END_TESTING\n");
}
