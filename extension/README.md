<!-- mdformat off(b/169948621#comment2) -->

# Extension of TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers is an experimental port of TensorFlow Lite designed to run machine learning models on microcontrollers and other low-power embedded devices with limited memory and computing resources, in a bare-metal environment. The TF Micro framework is still in its early stages and not all the operations of ML models are supported. The goal of the project is to implement some operations, commonly used in Neural Networks, that are still missing in TF Micro.

| Kernel | Documentation | Source |
| ------ | ------ | ------ |
| SQUEEZE | [API Doc](https://www.tensorflow.org/api_docs/python/tf/squeeze) | tensorflow/lite/micro/kernels/squeeze.cc |
| SPACE_TO_BATCH_ND | [API Doc](https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd) | tensorflow/lite/micro/kernels/space_to_batch_nd.cc |
| TRANSPOSE | [API Doc](https://www.tensorflow.org/api_docs/python/tf/transpose) | tensorflow/lite/micro/kernels/transpose.cc |
| ONE_HOT | [API Doc](https://www.tensorflow.org/api_docs/python/tf/one_hot) | tensorflow/lite/micro/kernels/one_hot.cc |

## Deploy to Microcontroller

The following instructions will help you build the sample for the STM32F7 using STM32CubeIDE.

Navigate into the tensorflow directory and run the Makefile in the TensorFlow Lite for Microcontrollers directory:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=stm32f4 TAGS=”CMSIS” generate_projects 
```

This will take a few minutes, so be patient. It generates a number of example projects and source code for you to use as a starting point.

You can find the source code at this location:

```
tensorflow/lite/micro/tools/make/gen/
```

In STM32CubeIDE, create a new STM32 project for your specific board. Make sure C++ is selected as the target language.

Copy model and TensorFlow Lite for Microcontroller source code files to the project:

```
cp extension/inc/* <project_directory>/Core/Inc
cp extension/src/* <project_directory>/Core/Src

mkdir <project_directory>/TFLM

cp -r tensorflow/lite/micro/tools/make/gen/*/prj/*hello_world*/make/* <project_directory>/TFLM

```

Delete the TensorFlow Lite for Microcontrollers examples folder, as it contains a template main.c application that we do not want in our project.

```
rm -r <project_directory>/TFLM/tensorflow/lite/micro/examples

```

Even though the source files are in our project, we still need to tell the IDE to include them in the build process. Go to Project > Properties. In that window, go to C/C++ General > Paths and Symbols > Includes tab > GNU C. Click Add. In the pop-up, click Workspace. Select the TFLM directory in your project. Check Add to all configurations and Add to all languages.

Repeat this process to add the following directories from TFLM/third_party to the build:

```
flatbuffers/include
gemmlowp
ruy
```

Head to the Source Location tab and add <project_directory>/TFLM to both the Debug and Release configurations.

Build your project and click Run > Debug. In the Debug perspective, click the Play/Pause button to begin running your code on the microcontroller.

Open a serial terminal to your development board, and you should see the output of inference (the baud rate is specified in the main.c file). On OSX and Linux, the following command should work, replacing /dev/tty.devicename with the name of your device as it appears in /dev:

```
screen /dev/tty.usbmodem14403 115200
```

To stop the scrolling, hit Ctrl+A, immediately followed by Esc. You can then use the arrow keys to explore the output, which will contain the results of implemented kernels tests.

To stop viewing the debug output with screen, hit Ctrl+A, immediately followed by the K key, then hit the Y key.
