#!/bin/bash

# sample command to run the model using adaptive learning rate 
python3 symnet.py --dataset "/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/X2_tr.npy,/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/Y2_tr.npy,/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/X_va.npy,/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/Y_va.npy" --batch-size 200 --epochs 1000 --flag-type "adaptive" 

# sample command to run the model using constant learning rate 
python3 symnet.py --dataset "/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/X2_tr.npy,/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/Y2_tr.npy,/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/X_va.npy,/home/surajubuntu/Documents/final_year_research_project/gene_dataset/GEO_tr_1/Y_va.npy" --batch-size 200 --epochs 1000 --flag-type "constant;0.1" 
