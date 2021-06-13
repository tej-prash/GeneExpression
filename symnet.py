import argparse
from symnet.regression import LipGeneModel

parser = argparse.ArgumentParser(
    description='Train a deep learning model on gene dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, required=True, help='The dataset to train from')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--train-split', type=float, default=0.7, help='Split to use for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--flag-type', type=str, required=True, help='Type of flag(adaptive/constant;LR)')

def main():
    #Fix weights

    args = parser.parse_args()
    bs = args.batch_size
    train_split = args.train_split
    n_epochs = args.epochs
    flag_type = args.flag_type

  
    model=LipGeneModel(args.dataset, bs=bs, train_size=train_split, epochs=n_epochs,optimizer='sgd',flag_type=flag_type)

    model.fit()
    loss, accuracy = model.score()
    
    print('Loss =', loss, '\nAccuracy =', accuracy)
    
if __name__ == '__main__':
    main()
