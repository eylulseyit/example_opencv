#!/bin/bash

cd build
rm -rf *
cmake ..
make -j4
cp opencv_example ../
cd ..
./opencv_example