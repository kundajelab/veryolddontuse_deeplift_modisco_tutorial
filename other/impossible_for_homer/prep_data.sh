#!/usr/bin/env bash

echo "generating the data"
./generate_synthetic_sequences.py
shuffle_corresponding_lines sequences.simdata --ignore_title
mv shuffled_sequences.simdata sequences.simdata

perl -ane 'if ($. > 1) {print ">$F[0]\n$F[1]\n"}' sequences.simdata > sequences.fa

echo "zipping up the files"
gzip -f sequences.simdata
gzip -f sequences.fa
