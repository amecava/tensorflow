/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"

namespace tflite {
namespace {

struct TransposeContext {
  TransposeContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    perm = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* perm;
  TfLiteTensor* output;
};

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                TransposeContext* op_context) {
  int dims = NumDimensions(op_context->input);
  const int32* perm_data = GetTensorData<int32_t>(op_context->perm);

  // Ensure validity of the permutations tensor as a 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->perm), 1);
  TF_LITE_ENSURE_EQ(context, op_context->perm->dims->data[0], dims);
  for (int idx = 0; idx < dims; ++idx) {
    TF_LITE_ENSURE_MSG(context, (perm_data[idx] >= 0 && perm_data[idx] < dims),
                       "Transpose op permutations array is out of bounds.");
  }

  // Checks output dimensions.
  TfLiteIntArray* input_size = op_context->input->dims;
  TfLiteIntArray* output_size = op_context->output->dims;
  for (int idx = 0; idx < dims; ++idx) {
	TF_LITE_ENSURE_EQ(context, output_size->data[idx], input_size->data[perm_data[idx]]);
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TransposeContext op_context(context, node);

  // Ensure validity of input tensor.
  TF_LITE_ENSURE_MSG(context, NumDimensions(op_context.input) <= 5,
                     "Transpose op only supports 1D-5D input arrays.");
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);

  return ResizeOutputTensor(context, &op_context);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TransposeContext op_context(context, node);

  const int32* perm_data = GetTensorData<int32_t>(op_context.perm);
  const int size = op_context.perm->dims->data[0];

  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

  #define TF_LITE_TRANSPOSE(scalar)                    \
    reference_ops::Transpose(						   \
       params, GetTensorShape(op_context.input),       \
       GetTensorData<scalar>(op_context.input),        \
       GetTensorShape(op_context.output),              \
       GetTensorData<scalar>(op_context.output));

  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (op_context.input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      TF_LITE_TRANSPOSE(int32_t);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      TF_LITE_TRANSPOSE(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_TRANSPOSE(int16_t);
      break;
    case kTfLiteInt64:
      TF_LITE_TRANSPOSE(int64_t);
      break;
    case kTfLiteBool:
      if (sizeof(bool) == 1) {
    	TF_LITE_TRANSPOSE(int8_t);
      } else {
        TF_LITE_TRANSPOSE(bool);
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported by Transpose.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_TRANSPOSE

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_TRANSPOSE() {
  return {/*init=*/nullptr,
		  /*free=*/nullptr,
		  /*prepare=*/Prepare,
		  /*invoke=*/Eval,
		  /*profiling_string=*/nullptr,
		  /*builtin_code=*/0,
		  /*custom_name=*/nullptr,
		  /*version=*/0};
}

}  // namespace tflite
