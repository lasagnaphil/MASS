#!/bin/bash

mkdir -p build
pushd build
cmake -DMASS_BUILD_RENDERER=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/home/$USER/pkgenv \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH=/home/$USER/pkgenv \
      -DFCL_INCLUDE_DIRS=/home/$USER/pkgenv/include/fcl \
      ..
make -j$(nproc)
popd
