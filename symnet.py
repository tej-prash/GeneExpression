import argparse

parser = argparse.ArgumentParser(
    description='Train a deep learning model on your dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--task', type=str, required=True, help='Type of task')
parser.add_argument('--dataset', type=str, required=True, help='The dataset to train from')
parser.add_argument('--data-type', type=str, required=True, help='Type of data')
parser.add_argument('--labels', type=str, required=True, help='The source of labels')


def main():
    args = parser.parse_args()


if __name__ == '__main__':
    main()
