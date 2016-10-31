#!/usr/bin/env bash
mkdir splits
zcat sequences.simdata.gz | perl -lane 'if ($. > 1) {if ($.%10!=0 and $.%10!=1) {print $F[0]}}' | gzip -c > splits/train.txt.gz
zcat sequences.simdata.gz | perl -lane 'if ($. > 1) {if ($.%10==0) {print $F[0]}}' | gzip -c > splits/valid.txt.gz
zcat sequences.simdata.gz | perl -lane 'if ($. > 1) {if ($.%10==1) {print $F[0]}}' | gzip -c > splits/test.txt.gz

