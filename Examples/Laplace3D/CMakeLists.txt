CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

set(DEAL_II_DIR "/u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/dealii/9.1.1" CACHE PATH "dealII installation path")

FIND_PACKAGE(deal.II 9.1.1 REQUIRED
  HINTS ${DEAL_II_DIR}
    )
DEAL_II_INITIALIZE_CACHED_VARIABLES()


SET(DEAL_II_CXX_FLAGS "-std=c++14 -Wno-deprecated-declarations -Wno-deprecated")
SET(DEAL_II_CXX_FLAGS_RELEASE " ")

set(CMAKE_BUILD_TYPE Release)

PROJECT(PDAS)

INCLUDE_DIRECTORIES(../../include ../../include/muParser)

ADD_EXECUTABLE(main main_Laplace_3D.cpp)
DEAL_II_SETUP_TARGET(main)
