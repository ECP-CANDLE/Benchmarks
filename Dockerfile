FROM ubuntu:22.04
RUN apt update -y 
RUN apt install -y git \
    python3 \
    python3-pip
WORKDIR /usr/repos
RUN pip3 install torch tensorflow-gpu
RUN git clone https://github.com/ECP-CANDLE/Benchmarks && cd Benchmarks && git checkout develop
RUN git clone https://github.com/ECP-CANDLE/candle_lib.git && cd candle_lib && git checkout develop && python3 setup.py install
ENV CANDLE_DATA_DIR=/data/
