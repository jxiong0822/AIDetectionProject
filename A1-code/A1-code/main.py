import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse


def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))


def read_data(path):
    train_frame = pd.read_csv(path + 'train.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        dev_frame = pd.read_csv(path + 'dev.csv')
        test_frame = pd.read_csv(path + 'test.csv')
    except:
        dev_frame = train_frame
        test_frame = train_frame

    return train_frame, dev_frame, test_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'BonusClassifier'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, dev_frame, test_frame = read_data(args.path)

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(train_frame['text'][i]))

    WordTable = feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']

    import random
    # form dev set for evaluation
    tokenized_text = []
    for i in range(0, len(dev_frame['text'])):
        tokenized_text.append(tokenize(dev_frame['text'][i]))
    X_dev = feat_extractor.transform_list(tokenized_text)
    Y_dev = dev_frame['label']
    
    # find random X_DEV
    
    num = random.randint(0, len(X_dev) - 10)
    X_dev = X_dev[num:num + 10]
    Y_dev = Y_dev[num:num + 10]
    # print(X_dev, Y_dev)
    # print("\n\n\n\n\n")
    # print(Y_dev)
    
    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']


    if args.model == "AlwaysPredictZero":
        model = AlwaysPredictZero()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()
    elif args.model == 'BonusClassifier':
        model = BonusClassifier()
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")

    start_time = time.time()
    model.fit(X_train,Y_train)
    
    print("===== Train Accuracy =====")
    maxtrain, mintrain, restrain = model.predict(X_train)
    accuracy(restrain, Y_train)
    
    print("===== Dev Accuracy =====")
    maxdev, mindev, resdev = model.predict(X_dev)
    accuracy(resdev, Y_dev)
    
    print("===== Test Accuracy =====")
    maxtest, mintest, restest = model.predict(X_test)
    accuracy(restest, Y_test)
    print("Max Ratios:\n")
    for ratio, element in maxtest:
        print(f"ratio is {ratio}, element is {list(WordTable.keys())[element]}\n")
    
    print("Min Ratios:\n")
    for ratio, element in mintest:
        print(f"ratio is {ratio}, element is {list(WordTable.keys())[element]}\n")
        
    
    print("Time for training, dev, and test: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()
