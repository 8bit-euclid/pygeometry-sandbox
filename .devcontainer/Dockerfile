FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install CMake 3.31.8 from official binaries (last 3.x version, compatible with cmake_minimum_required 3.1)
RUN wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz \
    && tar -xzf cmake.tar.gz -C /opt \
    && mv /opt/cmake-3.31.8-linux-x86_64 /opt/cmake \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest \
    && ln -s /opt/cmake/bin/cpack /usr/local/bin/cpack \
    && rm cmake.tar.gz
