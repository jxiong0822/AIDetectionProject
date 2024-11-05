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
    # parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
    #                     choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'BonusClassifier'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
    parser.add_argument('--path', type=str, default = './1b_benchmark', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, dev_frame, test_frame = read_data(args.path)

    X_train, Y_train = train_frame['text'], train_frame['label']
    X_dev, Y_dev = dev_frame['text'], dev_frame['label']
    X_test, Y_test = test_frame['text'], test_frame['label']

    # print(X_train, Y_train)
    # print(X_dev, Y_dev)
    # print(X_test, Y_test)

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
    
    feat_extractor.fit(X_train_data)
    p1 = feat_extractor.perplexity(X_train_data)
    print("Train Data: ", p1)
    p2 = feat_extractor.perplexity(X_dev_data)
    print("Dev Data: ", p2)
    p3 = feat_extractor.perplexity(X_test_data)
    print("Test Data: ", p3)
    
    # find random X_DEV
    
    #num = random.randint(0, len(X_dev) - 10)
   #X_dev = X_dev[num:num + 10]
    #Y_dev = Y_dev[num:num + 10]
    # print(X_dev, Y_dev)
    # print("\n\n\n\n\n")
    # print(Y_dev)

    # if args.model == "AlwaysPredictZero":
    #     model = AlwaysPredictZero()
    # elif args.model == "NaiveBayes":
    #     model = NaiveBayesClassifier()
    # elif args.model == "LogisticRegression":
    #     model = LogisticRegressionClassifier()
    # elif args.model == 'BonusClassifier':
    #     model = BonusClassifier()
    # else:
    #     raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")

    # start_time = time.time()
    # model.fit(X_train,Y_train)
    
    # print("===== Train Accuracy =====")
    # maxtrain, mintrain, restrain = model.predict(X_train)
    # accuracy(restrain, Y_train)
    
    # print("===== Dev Accuracy =====")
    # maxdev, mindev, resdev = model.predict(X_dev)
    # accuracy(resdev, Y_dev)
    
    # print("===== Test Accuracy =====")
    # maxtest, mintest, restest = model.predict(X_test)
    # accuracy(restest, Y_test)

    # if args.model == "NaiveBayes":
    #     print("Max Ratios:\n")
    #     for ratio, element in maxtest:
    #         print(f"ratio is {ratio}, element is {list(WordTable.keys())[element]}\n")
    
    #     print("Min Ratios:\n")
    #     for ratio, element in mintest:
    #         print(f"ratio is {ratio}, element is {list(WordTable.keys())[element]}\n")
        
    
    # print("Time for training, dev, and test: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()
