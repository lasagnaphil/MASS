#!/bin/bash

ENVDIR=/home/$USER/pkgenv

mkdir -p build
pushd build
cmake -DMASS_BUILD_RENDERER=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$ENVDIR \
      -DCMAKE_INSTALL_PREFIX=/home/$USER/pkgenv \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH=/home/$USER/pkgenv \
      ..

make -j$(nproc)
popd
