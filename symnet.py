import argparse
from symnet.numeric import NumericModel
from symnet.image import ResNet
from symnet.regression import RegressionModel

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
parser.add_argument('--no-augment', action='store_true', help='Do not augment data for image datasets')


def main():
    #Fix weights

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
    augment = not args.no_augment

    if task=='regression':
        model=RegressionModel(args.dataset, n_classes=num_classes, label_column=labels, task=task, header=has_header,
                             activation=activation, bs=bs, train_size=train_split, epochs=n_epochs, balance=False)

    elif args.data_type == 'numeric':
        model = NumericModel(args.dataset, n_classes=num_classes, label_column=labels, task=task, header=has_header,
                             activation=activation, bs=bs, train_size=train_split, epochs=n_epochs, balance=balance)
    elif args.data_type == 'image':
        # Default to ResNet110 v2
        model = ResNet(args.dataset, label_column=labels, header=has_header, augment_data=augment, n=12, version=2,
                       n_classes=num_classes, bs=bs, activation=activation)

    model.fit()
    loss, accuracy = model.score()
    print('Loss =', loss, '\nAccuracy =', accuracy)
    # model.plot_Kz()
    #Predict model
    # loss=model.calculate_loss(model.x_test,model.y_test)
    # print('Loss =', loss)

if __name__ == '__main__':
    main()
