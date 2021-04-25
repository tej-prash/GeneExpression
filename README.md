# SymNet
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2e9024b5c2ff44279f49ea5382244d09)](https://app.codacy.com/app/yrahul3910/symnet?utm_source=github.com&utm_medium=referral&utm_content=yrahul3910/symnet&utm_campaign=Badge_Grade_Dashboard)

SymNet is a deep learning pipeline with a focus on simplicity. Functionality is available through command-line options or as an API. The focus is
on simplicity and getting quick results.

## API Usage
### Numeric data
The `symnet.py` file shows how to use the API for multi-class classification
on a tabular (CSV) dataset. Start by creating a model:  

    model = NumericModel(csv_path, n_classes=3, label_column='target', task='classification')

Then, you can call `fit` and `predict` on the model, or find the loss and accuracy using
the `score` method.

## Image data
Image classifiers inherit from `AbstractImageClassificationModel`. Currently,
only ResNet is implemented. See `symnet.py` for example usage. Like
all models, you can call `fit`, `predict`, and `score`.

## CLI Usage
You can use the `symnet.py` file to run classification on a tabular dataset. The available options are:
*  `--task`: One of `'classification'` and `'regression'`
*  `--dataset`: The CSV dataset.
*  `--data-type`: As of now, only `'numeric'` and `'image'` are supported.
*  `--labels`: The CSV column with labels
*  `--num-classes`: Number of classes (for classification)
*  `--activation`: The activation to use. Any of `('relu', 'elu', 'selu', 'sigmoid', 'softmax', 'linear', 'sbaf', 'arelu', 'softplus)`
*  `--no-header`: Indicates that the CSV does not have a header row
*  `--batch-size`: The batch size to use
*  `--train-split`: The training data subset split size
*  `--epochs`: The number of epochs
*  `--no-balance`: Do not rebalance classes in classification problems
*  `--no-augment`: For image datasets, do not augment the data


   
## Todo
-  [ ]  Add DenseNet architecture
-  [ ]  Add support for text datasets
-  [ ]  Add support for image segmentation tasks
-  [ ]  Resize and normalize images
-  [ ]  For images, use LipschitzLR scheduler

