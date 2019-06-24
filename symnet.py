import argparse
from symnet.numeric import NumericModel

parser = argparse.ArgumentParser(
    description='Train a deep learning model on your dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--task', type=str, required=True, help='Type of task (classification)')
parser.add_argument('--dataset', type=str, required=True, help='The dataset to train from')
parser.add_argument('--data-type', type=str, required=True, help='Type of data')
parser.add_argument('--labels', type=str, required=True, help='The source of labels')
parser.add_argument('--num-classes', type=int, help='Number of classes in classification problems')
parser.add_argument('--activation', type=str, default='relu', help='Activation function')
parser.add_argument('--no-header', action='store_true', help='No header in the CSV file')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--train-split', type=float, default=0.7, help='Split to use for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--no-balance', action='store_true', help='Do not rebalance classes')


def main():
    args = parser.parse_args()

    num_classes = args.num_classes
    labels = args.labels
    task = args.task
    has_header = False if args.no_header else 0
    activation = args.activation
    bs = args.batch_size
    train_split = args.train_split
    n_epochs = args.epochs
    balance = not args.no_balance

    if args.data_type == 'numeric':
        model = NumericModel(args.dataset, n_classes=num_classes, label_column=labels, task=task, header=has_header,
                             activation=activation, bs=bs, train_size=train_split, epochs=n_epochs, balance=balance)

    model.fit()
    loss, accuracy = model.score()
    print('Loss =', loss, '\nAccuracy =', accuracy)


if __name__ == '__main__':
    main()
