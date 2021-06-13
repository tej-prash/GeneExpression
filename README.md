# LipGene

SymNet is a deep learning pipeline with a focus on simplicity. Functionality is available through command-line options. The focus is
on simplicity and getting quick results.

# Dataset


# Setup

### Requirements
* `python` 3.x
* `pip`
* [`virtualenv`](https://virtualenv.pypa.io/en/latest/)

### Create new virtual environment
The following command creates a new virtual environment named `lipGeneEnv` in the current directory.
```sh
$ virtualenv lipGeneEnv
```

### Activate virtual environment
```sh
# Windows (CMD.exe)
$ path\to\lipGeneEnv\Scripts\activate.bat
# Unix
$ source path/to/lipGeneEnv/bin/activate
```
### Install the required Python modules
```sh
$ (lipGeneEnv) pip install -r requirements.txt
```

### Setup ```base_path``` in GeneExpression directory
Execution logs and model weights are saved here
```
mkdir test_results
cd test_results
mkdir LipGeneModel
cd ..
```

### CLI Usage
You can use the `symnet.py` file to run regression on gene dataset. The available options are:
*  `--dataset`: The dataset path.
*  `--batch-size`: The batch size to use
*  `--train-split`: The training data subset split size
*  `--epochs`: The number of epochs
*  `--flag-type`: "adaptive" or "constant;0.1"

### Sample commands

Adaptive learning rate
```
python3 symnet.py --dataset "/home/gene_dataset/GEO_tr_1/X2_tr.npy,/home/gene_dataset/GEO_tr_1/Y2_tr.npy,/home/gene_dataset/GEO_tr_1/X_va.npy,/home/gene_dataset/GEO_tr_1/Y_va.npy" --batch-size 200 --epochs 1000 --flag-type "adaptive" 
```
Constant learning rate
```
python3 symnet.py --dataset "/home/gene_dataset/GEO_tr_1/X2_tr.npy,/home/gene_dataset/GEO_tr_1/Y2_tr.npy,/home/gene_dataset/GEO_tr_1/X_va.npy,/home/gene_dataset/GEO_tr_1/Y_va.npy" --batch-size 200 --epochs 1000 --flag-type "constant;0.1" 
```
