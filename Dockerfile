# SPDX-License-Identifier: Apache-2.0
#
# Build Docker image
# `$ sudo docker build --no-cache -t net-signn:latest .`
#
# Save and Load Docker image
# `$ sudo docker save net-signn:latest | gzip > net-signn.tar.gz`
# `$ sudo docker load < net-signn.tar.gz`
#
# Run Docker image
# `$ sudo docker run -it -v ${PWD}/workspace:/root/workspace --network=host net-signn`
# Optional (if using USRP B210)
# `$ lsusb | grep "Ettus Research LLC"`
# `$ --device=/dev/bus/usb/004/002`

# SPDX-License-Identifier: Apache-2.0
# Use an official ubuntu base image (Focal Fossa: 20.04)
FROM ubuntu:focal
MAINTAINER Steve Rhee "steve.h.rhee@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
# Update and install the following yum packages
RUN apt -y update && apt -y upgrade 
RUN apt -y install nano passwd git ssh net-tools iputils-ping
RUN apt -y install libboost-all-dev cmake-qt-gui build-essential cmake
RUN apt -y install python3-protobuf python3-matplotlib python3-pip
RUN apt -y install protobuf-compiler gnuradio gnuradio-dev gqrx-sdr
RUN apt -y install jupyter-notebook jupyter
RUN apt -y install swig liborc-0.4-dev pkg-config 
RUN apt -y install clang-format doxygen libcppunit-dev 
RUN apt -y install liblog4cpp5-dev libspdlog-dev
RUN apt -y autoremove && apt -y autoclean

# Update UHD installation with FPGA binaries
RUN uhd_images_downloader

# Clone signn and prep repository 
RUN cd /root/
RUN git clone https://gitlab.com/londonium6/signn.git
RUN chmod -R a+rw signn
RUN cd signn
RUN python3 -m pip install -r requirements3.txt
RUN cd utils/dataset/gnuradio_sim
RUN wget https://cloud.libre.space/s/rzS3QaXLY6BTN3x/download/source_material.tar.gz
RUN tar -xvzf source_material.tar.gz
RUN mv gutenberg_shakespeare.txt ../source_material/
RUN mv serial-s01-e01.wav ../source_material/

ENTRYPOINT ["/bin/bash","-l"]
