# Minimal ESP-IDF application used only to measure the LYRA model package on the ESP32-S3
# (ml/lyra/resources.py). It runs features -> model -> detector on a synthetic window.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(lyra_size_probe)
