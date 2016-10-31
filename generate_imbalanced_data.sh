#!/usr/bin/env bash

./generate_imbalanced_data.py
rm *_info.txt
gzip *.simdata
