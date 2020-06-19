#!/bin/bash

python3 symnet.py --dataset "../gene_dataset/sampled_data/X1_tr.npy,../gene_dataset/sampled_data/Y1_tr.npy,../gene_dataset/X_va.npy,../gene_dataset/Y_va.npy" --task regression --data-type numeric --num-classes "0" --batch-size 200 --epochs 500 --filet "b" --flag-type "constant;0.1"
