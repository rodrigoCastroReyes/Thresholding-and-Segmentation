cmake_minimum_required(VERSION 2.8)
project( DeteccionEsquinas )
find_package( OpenCV REQUIRED )
add_executable(DeteccionEsquinas main.cpp )
target_link_libraries( DeteccionEsquinas ${OpenCV_LIBS} )
