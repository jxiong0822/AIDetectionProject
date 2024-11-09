import pandas as pd
from utils import *
import numpy as np
import time
import argparse


def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))


def read_data(path):
    train_frame = pd.read_csv(path + '.train.tokens', delimiter='/n', names=['text', 'label'])

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    dev_frame = pd.read_csv(path + '.dev.tokens', delimiter='/n', names=['text', 'label'])

    # train_frame = train_frame._append(dev_frame, ignore_index=True)
    try:
        test_frame = pd.read_csv(path + '.test.tokens', delimiter='/n', names=['text', 'label'])
    except:
        dev_frame = train_frame
        test_frame = train_frame

    return train_frame, dev_frame, test_frame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
    parser.add_argument('--path', type=str, default = './1b_benchmark', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, dev_frame, test_frame = read_data(args.path)

    X_train, Y_train = train_frame['text'], train_frame['label']
    X_dev, Y_dev = dev_frame['text'], dev_frame['label']
    X_test, Y_test = test_frame['text'], test_frame['label']

    X_train_data = [text.split() for text in X_train]
    X_dev_data = [text.split() for text in X_dev]
    X_test_data = [text.split() for text in X_test]

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "trigram":
        feat_extractor = Trigram()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")
    
    a = 0
    feat_extractor.fit(X_train_data)
    feat_extractor.MLE(a)
    p1 = feat_extractor.perplexity(X_train_data, a)
    print("Train Data: ", p1)
    p2 = feat_extractor.perplexity(X_dev_data, a)
    print("Dev Data: ", p2)
    p3 = feat_extractor.perplexity(X_test_data, a)
    print("Test Data: ", p3)

    unigram = UnigramFeature()
    unigram.fit(X_train_data)
    bigram = BigramFeature()
    bigram.fit(X_train_data)
    trigram = Trigram()
    trigram.fit(X_train_data)

    print("interpolation")
    ip1 = interpolation(unigram, bigram, trigram, X_train_data, 0.1, 0.3, 0.6) 
    print("Train Data: ", ip1)
    ip2 = interpolation(unigram, bigram, trigram, X_dev_data, 0.1, 0.3, 0.6) 
    print("Dev Data: ", ip2)
    ip3 = interpolation(unigram, bigram, trigram, X_dev_data, 0.1, 0.3, 0.6) 
    print("Test Data: ", ip3)
    

if __name__ == '__main__':
    main()
