#!/bin/bash

python3 symnet.py --dataset "../gene_dataset/sampling_with_replacement/X2_tr.npy,../gene_dataset/sampling_with_replacement/Y2_tr.npy,../gene_dataset/X_va.npy,../gene_dataset/Y_va.npy" --task regression --data-type numeric --num-classes "0" --batch-size 200 --epochs 500 --filet "b" --flag-type "constant;0.1"