export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH
export CC=/opt/compiler/gcc-4.8.2/bin/gcc
export CXX=/opt/compiler/gcc-4.8.2/bin/g++
export LD_LIBRARY_PATH=/home/work/cuda-10.2/lib64/:/home/work/cudnn/cudnn_v8.0.1/cuda/lib64:$LD_LIBRARY_PATH
cmake .. -DTENSORRT_ROOT=/home/work/TensorRT-7.1.3.4/ -DPYTHON_INCLUDE=/home/work/python/python2_7_15/include \
-DTARGET_FILE=/usr/local/include/ -Dpybind11_DIR=/home/work/python/python2_7_15/lib/python2.7/site-packages/pybind11 \
-DGPU_ARCHS="61" -DProtobuf_INCLUDE_DIR=/usr/local/include/ -DProtobuf_LIBRARY=/usr/local/lib/libprotobuf.so \
-DCUDNN_LIBRARY=/home/work/cudnn/cudnn_v8.0.1/cuda/lib64  -DCUDNN_ROOT_DIR=/home/work/cudnn/cudnn_v8.0.1/cuda/ \
-DCUDA_INCLUDE_DIRS=/home/work/cuda-10.2/include/ \
-DBUILD_ONNX_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=/home/work/sunminqi/env/myonnxtrt/build/
