FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install dependencies
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv - Python package and project manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Install CMake 3.31.8 from official binaries (last 3.x version, compatible with cmake_minimum_required 3.1)
RUN wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz \
    && tar -xzf cmake.tar.gz -C /opt \
    && mv /opt/cmake-3.31.8-linux-x86_64 /opt/cmake \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest \
    && ln -s /opt/cmake/bin/cpack /usr/local/bin/cpack \
    && rm cmake.tar.gz

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
