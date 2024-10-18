import numpy as np

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
        # Add your code here!
        pass

    def fit(self, X, Y):
        # Add your code here!
        pass

    def predict(self, X):
        # Add your code here!
        pass

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        self.learning_rate = 0.15
        self.epochs = 5000
        self.weights = None
        self.bias = None        

    def fit(self, X, Y):
        # Add your code here!
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-z))

            dw = 1/X.shape[0] * np.dot(X.T, (y_pred - Y)) 
            db = 1/X.shape[0] * np.sum(y_pred - Y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = -1/X.shape[0] * np.sum(Y * np.log(y_pred) + (1-Y) * np.log(1-y_pred))
            print(f'Epoch {epoch+1}, Loss: {loss}')
        return self.weights, self.bias
        
    
    def predict(self, X):
        # Add your code here!
        z = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-z))
        result = [1 if i > 0.5 else 0 for i in y_pred]
        return result


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
