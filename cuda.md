```sh
python -m torch.utils.collect_env
```

# NVIDIA-Driver
```sh
nvidia-smi
```

# CUDA Toolkit
```sh
nvcc -V
```

# cuDNN
```sh
### Check version.
ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
### The result would look like,
# libcudnn_adv.so.9 -> libcudnn_adv.so.9.0.0
# libcudnn_engines_runtime_compiled.so.9 -> libcudnn_engines_runtime_compiled.so.9.0.0
# libcudnn_graph.so.9 -> libcudnn_graph.so.9.0.0
# libcudnn_ops.so.9 -> libcudnn_ops.so.9.0.0
# libcudnn_cnn.so.9 -> libcudnn_cnn.so.9.0.0
# libcudnn_engines_precompiled.so.9 -> libcudnn_engines_precompiled.so.9.0.0
# libcudnn_heuristic.so.9 -> libcudnn_heuristic.so.9.0.0
# libcudnn.so.9 -> libcudnn.so.9.0.0
```

# OpenCV
```sh
mkdir opencv && cd opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
unzip opencv.zip 
unzip opencv_contrib.zip

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_PACKAGE=OFF -D BUILD_EXAMPLES=OFF -D WITH_TBB=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_CUFFT=ON -D WITH_NVCUVID=ON -D WITH_IPP=OFF -D WITH_V4L=OFF -D WITH_LIBV4L=ON -D WITH_1394=OFF -D WITH_GTK=ON -D WITH_QT=OFF -D WITH_OPENGL=OFF -D WITH_EIGEN=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D BUILD_JAVA=OFF -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D OPENCV_SKIP_PYTHON_LOADER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=/home/jbkim/Downloads/opencv/opencv/opencv_contrib-4.5.5/modules -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D CUDA_ARCH_BIN=8.6 -D CUDA_ARCH_PTX=8.6 -D CUDNN_LIBRARY=/usr/local/cuda-12.2/lib64/libcudnn.so.9.0.0 -D CUDNN_INCLUDE_DIR=/usr/local/cuda-12.2/include -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.8/dist-packages ..
```

# TensorRT
- References:
    - https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/0.%20Running%20This%20Guide.ipynb
```sh
pip install --upgrade --index-url https://pypi.ngc.nvidia.com nvidia-tensorrt
```