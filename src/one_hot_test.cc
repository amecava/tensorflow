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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

void one_hot_test(tflite::ErrorReporter* error_reporter) {

  TF_LITE_REPORT_ERROR(error_reporter, "TestOneHotOp");

  // Parameter

  const int input_dims_data[] = {1, 3};
  const int output_dims_data[] = {2, 3, 3};
  const int32_t input_data[] = {0, 1, 2};

  const int depth_dims_data[] = {1, 1};
  const int32_t depth_data[] = {3};
  const int on_dims_data[] = {1, 1};
  const int32_t on_data[] = {1};
  const int off_dims_data[] = {1, 1};
  const int32_t off_data[] = {0};

  TfLiteOneHotParams params = {-1};

  const int expected_output_size = 9;
  const int32_t expected_output[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  int32_t output_data[9];

  // Tensor

  TfLiteIntArray* input_dims1 = tflite::testing::IntArrayFromInts(input_dims_data);
  TfLiteIntArray* depth_dims1 = tflite::testing::IntArrayFromInts(depth_dims_data);
  TfLiteIntArray* on_dims1 = tflite::testing::IntArrayFromInts(on_dims_data);
  TfLiteIntArray* off_dims1 = tflite::testing::IntArrayFromInts(off_dims_data);
  TfLiteIntArray* output_dims1 = tflite::testing::IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  TfLiteTensor tensors[tensors_size];
  tensors[0] = tflite::testing::CreateTensor(input_data, input_dims1);
  tensors[1] = tflite::testing::CreateTensor(depth_data, depth_dims1);
  tensors[2] = tflite::testing::CreateTensor(on_data, on_dims1);
  tensors[3] = tflite::testing::CreateTensor(off_data, off_dims1);
  tensors[4] = tflite::testing::CreateTensor(output_data, output_dims1);

  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = tflite::testing::IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = tflite::testing::IntArrayFromInts(outputs_array_data);

  // Register

  const TfLiteRegistration registration = tflite::Register_ONE_HOT();
  tflite::micro::KernelRunner runner(
         registration, tensors, tensors_size, inputs_array, outputs_array,
         reinterpret_cast<void*>(&params), error_reporter);

  // Init

  const char* init_data = reinterpret_cast<const char*>(&params);

  TfLiteStatus runner_status = runner.InitAndPrepare(init_data);
  if (runner_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "  Init: failed");
    TF_LITE_REPORT_ERROR(error_reporter, "  Prepare: failed");
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "  Init: ok");
  TF_LITE_REPORT_ERROR(error_reporter, "  Prepare: ok");

  // Invoke

  TfLiteStatus invoke_status = runner.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "  Invoke: failed");
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "  Invoke: ok");

  // Eval

  bool test_status = true;
  for (int i = 0; i < expected_output_size; ++i) {
  	if (output_data[i] != expected_output[i]) {
  	  test_status = false;
  	}
  }

  if (test_status) {
    TF_LITE_REPORT_ERROR(error_reporter, "  Test: PASSED");
  } else {
	TF_LITE_REPORT_ERROR(error_reporter, "  Test: FAILED");
  }
}
