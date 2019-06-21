# SymNet
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2e9024b5c2ff44279f49ea5382244d09)](https://app.codacy.com/app/yrahul3910/symnet?utm_source=github.com&utm_medium=referral&utm_content=yrahul3910/symnet&utm_campaign=Badge_Grade_Dashboard)

SymNet is a deep learning pipeline with a focus on simplicity. Functionality is available through command-line options or as an API. The focus is
on simplicity and getting quick results.

## Cite our work
SymNet uses the LipschitzLR learning rate policy: [arXiv:1902.07399](https://arxiv.org/abs/1902.07399)

BibTeX entry:  

    @article{yedida2019novel,
      title={A novel adaptive learning rate scheduler for deep neural networks},
      author={Yedida, Rahul and Saha, Snehanshu},
      journal={arXiv preprint arXiv:1902.07399},
      year={2019}
    }

## API Usage
### Numeric data
The `symnet.py` file shows how to use the API for multi-class classification
on a tabular (CSV) dataset. Start by creating a model:  

    model = NumericModel(csv_path, n_classes=3, label_column='target', task='classification')

Then, you can call `fit` and `predict` on the model, or find the loss and accuracy using
the `score` method. Currently, only classification is supported, but more features will
be added soon.

## CLI Usage
You can use the `symnet.py` file to run classification on a tabular dataset like this:  

    python3 symnet.py --task classification --dataset data.csv --data-type numeric --labels Y --num-classes 3 

## Todo
- [ ] Implement the SBAF and A-ReLU activations
- [ ] Add regression support
- [ ] Add support for image datasets
- [ ] Add support for text datasets
- [ ] Add support for image segmentation tasks