#!/usr/bin/en sh
g++ -I/home/hans/ncnn-20180516/src -I/home/hans/ncnn-20180516/build/src -include"./PCN.h" -O0 -g3 -Wall -c -fmessage-length=0  -std=c++11 -MMD -MP -MF"video.d" -MT"video.o" -o "video.o" "./video.cpp"
g++ -fopenmp -o "ncnn_PCN"  ./video.o  /home/hans/ncnn-20180516/build/src/libncnn.a -lopencv_contrib -lopencv_core -lopencv_highgui -lopencv_imgproc