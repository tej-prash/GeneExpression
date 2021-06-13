# LipGeneNet

LipGeneNet is a deep learning pipeline used for testing the Lipschitz Adaptive Learning Rate (LALR) for the task of Gene Expression Inference.

# Dataset
Gene Omnibus Expression(GEO) Dataset, based on the Affymetric Microarray platform, is a publicly available dataset prepared by the Broad Institute. It consists of 129 158 gene expression profiles, each profile consisting of 978 landmark genes and 21290 target genes. Chen et al. [1] performed joint quantile normalization and removed duplicates using two other Gene Expression datasets-Genotype-Tissue Expression (GTEx) and 1000 Genomes Expression Data (1000G).  The GTEx and 1000G datasets are based on the Illumina RNA-Seq platform and used Gencode V12 annotations to measure the expression levels of each gene [2], [3]. The modified GEO dataset obtained consists of over 90,000 gene expression profiles, each profile consisting of 943 landmark genes and 9520 target genes, standardized to zero mean and unit standard deviation for each gene. The modified GEO dataset was used for training by Chen et al. [2] and we use the same dataset for our experimentation.


## Instructions to deploy the LipGene model

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
*  `--flag-type`: Use "adaptive" or "constant;learning rate value"

### Sample commands

Adaptive learning rate
```
python3 symnet.py --dataset "/home/gene_dataset/GEO_tr_1/X2_tr.npy,/home/gene_dataset/GEO_tr_1/Y2_tr.npy,/home/gene_dataset/GEO_tr_1/X_va.npy,/home/gene_dataset/GEO_tr_1/Y_va.npy" --batch-size 200 --epochs 1000 --flag-type "adaptive" 
```
Constant learning rate
```
python3 symnet.py --dataset "/home/gene_dataset/GEO_tr_1/X2_tr.npy,/home/gene_dataset/GEO_tr_1/Y2_tr.npy,/home/gene_dataset/GEO_tr_1/X_va.npy,/home/gene_dataset/GEO_tr_1/Y_va.npy" --batch-size 200 --epochs 1000 --flag-type "constant;0.1" 
```

## References
<a id="1">[1]</a> 
Y.  Chen,  Y.  Li,  R.  Narayan,  A.  Subramanian,  and  X.  Xie,  “Geneexpression  inference  with  deep  learning,”Bioinformatics,  vol.  32,no. 12, pp. 1832–1839, 2016
<a id="2">[2]</a> 
G. Consortiumet al., “The genotype-tissue expression (gtex) pilotanalysis: multitissue gene regulation in humans,”Science, vol. 348,no. 6235, pp. 648–660, 2015.
<a id="3">[3]</a> 
T.  Lappalainen,  M.  Sammeth,  M.  R.  Friedl ̈ander,  P.  AC‘t  Hoen,J.   Monlong,   M.   A.   Rivas,   M.   Gonzalez-Porta,   N.   Kurbatova,T.   Griebel,   P.   G.   Ferreiraet   al.,   “Transcriptome   and   genomesequencing uncovers functional variation in humans,”Nature, vol.501, no. 7468, pp. 506–511, 2013
