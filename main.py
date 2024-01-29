import config as c
import argparse
from train import train
from utils import load_datasets, make_dataloaders


def main():

    print(f' step 1. loading dataset')
    dataset_path = args.dataset_path
    class_name = args.class_name
    train_set, test_set = load_datasets(dataset_path, class_name)
    train_loader, test_loader = make_dataloaders(train_set, test_set)

    print(f' step 2. make model')
    #model = train(train_loader, test_loader)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_path', type=str, default='dummy_dataset')
    args.add_argument('--class_name', type=str, default='dummy_class')
    main(args)