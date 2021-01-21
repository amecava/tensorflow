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

void transpose_test(tflite::ErrorReporter* error_reporter) {

  TF_LITE_REPORT_ERROR(error_reporter, "TestTransposeOp");

  // Parameter

  const int input_dims_data[] = {3, 2, 3, 4};
  const int output_dims_data[] = {3, 4, 2, 3};
  const int32_t input_data[] = {1,  2,  3,  4,  5,  6,  7,  8,
			                  9,  10, 11, 12, 13, 14, 15, 16,
			                  17, 18, 19, 20, 21, 22, 23, 24};

  const int perm_dims_data[] = {1, 3};
  const int32_t perm_data[] = {2, 0, 1};

  tflite::TransposeParams params = {};

  const int expected_output_size = 24;
  const int expected_output[] = {1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22,
                                 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24};

  int32_t output_data[24];

  // Tensor

  TfLiteIntArray* input_dims1 = tflite::testing::IntArrayFromInts(input_dims_data);
  TfLiteIntArray* perm_dims1 = tflite::testing::IntArrayFromInts(perm_dims_data);
  TfLiteIntArray* output_dims1 = tflite::testing::IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  TfLiteTensor tensors[tensors_size];
  tensors[0] = tflite::testing::CreateTensor(input_data, input_dims1);
  tensors[1] = tflite::testing::CreateTensor(perm_data, perm_dims1);
  tensors[2] = tflite::testing::CreateTensor(output_data, output_dims1);

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = tflite::testing::IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = tflite::testing::IntArrayFromInts(outputs_array_data);

  // Register

  const TfLiteRegistration registration = tflite::Register_TRANSPOSE();
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
