#!/usr/bin/en sh
g++ -I/home/hans/ncnn-20180516/src -I/home/hans/ncnn-20180516/build/src -include"./PCN.h" -O0 -g3 -Wall -c -fmessage-length=0  -std=c++11 -MMD -MP -MF"cam.d" -MT"cam.o" -o "cam.o" "./cam.cpp"
g++ -fopenmp -o "ncnn_PCN_cam"  ./cam.o  /home/hans/ncnn-20180516/build/src/libncnn.a -lopencv_contrib -lopencv_core -lopencv_highgui -lopencv_imgproc
