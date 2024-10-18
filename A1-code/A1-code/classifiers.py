import numpy as np
import math

# You need to build your own model here instead of using existing Python
# packages such as sklearn!

## But you may want to try these for comparison, that's fine.
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
              is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where
              N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
            is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the
            number of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Pos/Neg total_counts holds the total number of features seen in pos/neg
        # i.e sum(whole pos array)
        self.pos_total_counts = 0
        self.neg_total_counts = 0
        # feature counts will have an array of 
        self.pos_features = []
        self.neg_features = []
        # Total counts of samples
        self.total_samples = 0
        

    def fit(self, X, Y):
        self.Y = list(Y)
        self.pos_features = [0] * len(X[0])
        self.neg_features = [0] * len(X[0])
        # print(X)
        # print(Y)
        # raise Exception("Must be implemented")
        for row in range(len(Y)):
            # if positive review
            if Y[row] == 1:
                for vals in range(len(X[row])):
                    self.pos_features[vals] += int(X[row][vals] + 1)
                    self.pos_total_counts += int(X[row][vals] + 1)
                    self.total_samples += int(X[row][vals] + 1)
            else:
                for vals in range(len(X[row])):
                    self.neg_features[vals] += int(X[row][vals] + 1)
                    self.neg_total_counts += int(X[row][vals] + 1)
                    self.total_samples += int(X[row][vals] + 1)
                    
        #print(self.neg_features)
        #print("\n\n")
        #print(self.pos_features)
        # print("\n\n\n\n\n")
    
    def predict(self, X):
        """Predict class labels for the given test data X"""

        # X is a list of feature sets (each element contains the features of a single sample)
        output = []
        ratios = set()
        for row in range(len(X)):
            pos_predict_count = 0
            neg_predict_count = 0
            # each row is a guess we need to do
            for element in range(len(X[row])):
                # if element is greater than 0, it mean we see a keyword in the review
                if X[row][element] > 0:
                    pos_temp = math.log2((self.pos_features[element] / self.pos_total_counts * (self.Y.count(1))/len(self.Y)) / ((self.pos_features[element] + self.neg_features[element]) / self.total_samples))
                    pos_predict_count += pos_temp
                    neg_temp = math.log2((self.neg_features[element] / self.neg_total_counts * (self.Y.count(0))/len(self.Y)) / ((self.pos_features[element] + self.neg_features[element]) / self.total_samples))
                    neg_predict_count += neg_temp
                    ratio = pos_temp/neg_temp
                    tup = (ratio, element)
                    # ratios.append(tup)
                    ratios.add(tup)
                    
                    # (+/-_features * 0.5) / (+/-_features * +/-_total_count)/total_samples
            #print("pos is %s, neg is %s\n", pos_predict_count, neg_predict_count)
            output.append(1 if pos_predict_count > neg_predict_count else 0)
        # print(output)
        ratios = sorted(ratios)
        max_ratios = ratios[len(ratios) - 10:]
        min_ratios = ratios[:10]
        #for elem in min_ratios:
            #print(f"yo, ratio is {elem[0]}, index is {elem[1]}")
        return min_ratios, max_ratios, output   
    
# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Pos/Neg total_counts holds the total number of features seen in pos/neg
        # i.e sum(whole pos array)
        self.pos_total_counts = 0
        self.neg_total_counts = 0
        # feature counts will have an array of 
        self.pos_features = []
        self.neg_features = []
        # Total counts of samples
        self.total_samples = 0
        

    def fit(self, X, Y):
        self.Y = list(Y)
        self.pos_features = [0] * len(X[0])
        self.neg_features = [0] * len(X[0])
        # print(X)
        # print(Y)
        # raise Exception("Must be implemented")
        for row in range(len(Y)):
            # if positive review
            if Y[row] == 1:
                for vals in range(len(X[row])):
                    self.pos_features[vals] += int(X[row][vals] + 1)
                    self.pos_total_counts += int(X[row][vals] + 1)
                    self.total_samples += int(X[row][vals] + 1)
            else:
                for vals in range(len(X[row])):
                    self.neg_features[vals] += int(X[row][vals] + 1)
                    self.neg_total_counts += int(X[row][vals] + 1)
                    self.total_samples += int(X[row][vals] + 1)
                    
        #print(self.neg_features)
        #print("\n\n")
        #print(self.pos_features)
        # print("\n\n\n\n\n")
    
    def predict(self, X):
        """Predict class labels for the given test data X"""

        # X is a list of feature sets (each element contains the features of a single sample)
        output = []
        ratios = set()
        for row in range(len(X)):
            pos_predict_count = 0
            neg_predict_count = 0
            # each row is a guess we need to do
            for element in range(len(X[row])):
                # if element is greater than 0, it mean we see a keyword in the review
                if X[row][element] > 0:
                    pos_temp = math.log2((self.pos_features[element] / self.pos_total_counts * (self.Y.count(1))/len(self.Y)) / ((self.pos_features[element] + self.neg_features[element]) / self.total_samples))
                    pos_predict_count += pos_temp
                    neg_temp = math.log2((self.neg_features[element] / self.neg_total_counts * (self.Y.count(0))/len(self.Y)) / ((self.pos_features[element] + self.neg_features[element]) / self.total_samples))
                    neg_predict_count += neg_temp
                    ratio = pos_temp/neg_temp
                    tup = (ratio, element)
                    # ratios.append(tup)
                    ratios.add(tup)
                    
                    # (+/-_features * 0.5) / (+/-_features * +/-_total_count)/total_samples
            #print("pos is %s, neg is %s\n", pos_predict_count, neg_predict_count)
            output.append(1 if pos_predict_count > neg_predict_count else 0)
        # print(output)
        ratios = sorted(ratios)
        max_ratios = ratios[len(ratios) - 10:]
        min_ratios = ratios[:10]
        """
        for elem in min_ratios:
            print(f"yo, ratio is {elem[0]}, index is {elem[1]}")
        """
        return min_ratios, max_ratios, output   
