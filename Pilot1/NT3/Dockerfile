FROM jdacs4c/improve-pytorch:0.0.1
# FROM ubuntu:22.04 
RUN apt update -y 
# RUN apt install -y git \
    # python3 \
    # python3-pip
WORKDIR /
# RUN pip3 install torch tensorflow-gpu
RUN pip3 install tensorflow-gpu

RUN git clone https://github.com/ECP-CANDLE/Benchmarks && cd Benchmarks && git checkout develop
RUN git clone https://github.com/ECP-CANDLE/candle_lib.git && cd candle_lib && git checkout develop && python3 setup.py install
ENV CANDLE_DATA_DIR=/data/
ENV REPO_DIR=/Benchmarks
# ASSUMING BUILD IS SPECIFIC BENCHMARK DIRECTORY, e.g. Pilot1/NT3
COPY *.sh /usr/local/bin/



