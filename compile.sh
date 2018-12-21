#!/usr/bin/en sh
g++ -I/home/hans/ncnn-20180516/src -I/home/hans/ncnn-20180516/build/src -include"./PCN.h" -O3 -c -std=c++11 -fPIC -o "cam.o" "./cam.cpp"
g++ -fopenmp -o "ncnn_PCN_cam"  ./cam.o  /home/hans/ncnn-20180516/build/src/libncnn.a -lopencv_core -lopencv_highgui -lopencv_imgproc
