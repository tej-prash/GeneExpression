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


def main():
    args = parser.parse_args()

    if args.data_type == 'numeric':
        model = NumericModel(args.dataset, n_classes=args.num_classes, label_column=args.labels, task=args.task)

    model.fit()
    loss, accuracy = model.score()
    print('Loss =', loss, '\nAccuracy =', accuracy)


if __name__ == '__main__':
    main()
