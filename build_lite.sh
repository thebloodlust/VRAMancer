#!/bin/bash
set -e
echo "Build archive VRAMancer Lite (CLI only)"
make -f Makefile.lite lite
cd dist_lite
zip -r ../vramancer_lite.zip .
echo "Archive vramancer_lite.zip créée !"