FROM nvcr.io/nvidia/l4t-ml:r32.4.4-py3
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python3 setup.py install --plugins
RUN pip3 install tqdm cython pycocotools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt-get install -y python3-opencv python3-matplotlib python3-sklearn python3-sklearn-lib python3-tk
RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose && cd trt_pose && python3 setup.py install
RUN pip3 install --upgrade pip
RUN pip3 install jupyterlab scikit-learn==0.22.2.post1 pandas
