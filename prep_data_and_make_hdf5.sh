#!/usr/bin/env bash

./prep_data.sh

echo "splitting into train/test/valid"
./prep_splits.sh

echo "preparing hdf5 files"
make_hdf5 --yaml_configs make_hdf5_yaml/* --output_dir .
