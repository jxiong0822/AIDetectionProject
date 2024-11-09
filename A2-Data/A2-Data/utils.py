from collections import Counter, defaultdict
import math
import copy

class UnigramFeature():
    """Implementation of unigram feature extraction with MLE and perplexity calculation"""
    def __init__(self):
        self.unigram_counts = {}

    def fit(self, text_set: list):
        self.data = copy.deepcopy(text_set)
        for sentence in self.data:
            sentence.append('<STOP>')
            sentence.insert(0, '<START>')
        all_words = [word for text in self.data for word in text]

        self.all_words_freq = Counter(all_words)

        processed_data = []
        for word in all_words:
            if self.all_words_freq[word] < 3:
                processed_data.append('<UNK>')
            else:
                processed_data.append(word)
        
        for word in processed_data:
            if word != '<START>':
                if word not in self.unigram_counts:
                    self.unigram_counts[word] = 1
                else:
                    self.unigram_counts[word] += 1
        
        self.total_words = sum(self.unigram_counts.values())
        self.vocab_size = len(self.unigram_counts)
        self.vocab = set(self.unigram_counts.keys())

        return self.unigram_counts
        

    def MLE(self, alpha):
        self.prob = {}
        for word in self.unigram_counts:
            self.prob[word] = (self.unigram_counts[word] + alpha) / (self.total_words + (alpha * self.vocab_size)) #P(w) = (count(w) + alpha) / (total_words + alpha * vocab_size)
        
        self.prob['<UNK>'] = (alpha) / (self.total_words + (alpha * self.vocab_size)) #P(UNK) = alpha / (total_words + alpha * vocab_size)

        return self.prob

    def perplexity(self, test_data, alpha):
        self.test_data = copy.deepcopy(test_data)

        for sentence in self.test_data:
            sentence.append('<STOP>')
        
        self.test_words = [word for text in self.test_data for word in text]
        total_words = len(self.test_words)

        probs = self.MLE(alpha)
        log_prob = 0

        for word in self.test_words:
            if word not in probs.keys():
                word = '<UNK>'
            prob = probs[word]
            if prob > 0:
                log_prob += math.log2(prob)
        
        perplexity = 2**(-log_prob / total_words) #perplexity = 2^-(sum(log2(prob)) / total_words)

        return perplexity
            
class BigramFeature:
    """Bigram feature extractor analogous to the unigram one."""
    def __init__(self):
        self.bigram_counts = {}
        self.unigram_counts = {}
        self.vocab = set()

    def fit(self, text_set: list):
        self.data = copy.deepcopy(text_set)

        for sentence in self.data:
            sentence.append('<STOP>')
            sentence.insert(0, '<START>')

        all_words = [word for text in self.data for word in text]

        self.all_words_freq = Counter(all_words)

        processed_data = []
        for word in all_words:
            if self.all_words_freq[word] < 3:
                processed_data.append('<UNK>')
            else:
                processed_data.append(word)
        
        for word in processed_data:
            if word not in self.unigram_counts:
                self.unigram_counts[word] = 1
            else:
                self.unigram_counts[word] += 1
        
        
        self.vocab = set(self.unigram_counts.keys())
        self.vocab_size = len(self.vocab)
        self.total_words = sum(self.unigram_counts.values())

        for i in range(1, len(processed_data)):
            bigram = (processed_data[i-1], processed_data[i])
            if processed_data[i-1] == '<STOP>' and processed_data[i] == '<START>':
                continue
            if bigram not in self.bigram_counts:
                self.bigram_counts[bigram] = 1
            else:
                self.bigram_counts[bigram] += 1

        # print("(HDTV, .)", self.bigram_counts[('HDTV', '.')])
        # print("HDTV", self.unigram_counts['HDTV'])
        return self.bigram_counts

    def MLE(self, alpha):
        """Calculate MLE with smoothing."""
        # print(self.vocab_size)
        self.probs = {}
        for bigram in self.bigram_counts:
            probablity = (self.bigram_counts[bigram] + alpha) / (self.unigram_counts[bigram[0]] + (alpha * self.vocab_size))
            self.probs[bigram] = probablity
        
        self.probs[('<UNK>', bigram[1])] = alpha / (self.unigram_counts[bigram[0]] + (alpha * self.vocab_size))
        self.probs[(bigram[0], '<UNK>')] = alpha / (self.unigram_counts[bigram[0]] + (alpha * self.vocab_size))
        self.probs[('<UNK>', '<UNK>')] = alpha / (self.unigram_counts[bigram[0]] + (alpha * self.vocab_size))

        return self.probs

    def perplexity(self, test_data, alpha):
        """Calculate perplexity on the test data."""
        # Preprocess test data by adding <START> and <STOP> and replacing rare words with <UNK>
        self.test_data = copy.deepcopy(test_data)

        for sentence in self.test_data:
            sentence.append('<STOP>')
            sentence.insert(0, '<START>')
        
        self.test_words = [word for text in self.test_data for word in text]
        total_words = len(self.test_words) - len(self.test_data)

        probs = self.MLE(alpha)
        log_prob = 0

        for i in range(1, len(self.test_words)):
            if self.test_words[i] not in self.vocab:
                self.test_words[i] = '<UNK>'
            if self.test_words[i-1] not in self.vocab:
                self.test_words[i-1] = '<UNK>'

            bigram = (self.test_words[i-1], self.test_words[i])
            prob = probs.get(bigram, 0)
            # print(bigram, prob)
            if prob > 0:
                log_prob += math.log2(prob)

        perplexity = 2**(-log_prob / total_words)

        return perplexity

            
class Trigram:
    def __init__(self):
        self.trigram = {}
        self.trigram_counts = {}
        self.bigram_counts = {}
        self.vocab = set()
        
    def fit(self, text_set: list):
        self.data = copy.deepcopy(text_set)

        for sentence in self.data:
            sentence.append('<STOP>')
            sentence.insert(0, '<START>')
            sentence.insert(0, '<START>')

        all_words = [word for text in self.data for word in text]

        self.all_words_freq = Counter(all_words)

        processed_data = []
        for word in all_words:
            if self.all_words_freq[word] < 3:
                processed_data.append('<UNK>')
            else:
                processed_data.append(word)

        for word in processed_data:
            if word not in self.vocab:
                self.vocab.add(word)

        for i in range(1, len(processed_data)):
            bigram = (processed_data[i-1], processed_data[i])
            if bigram not in self.bigram_counts:
                self.bigram_counts[bigram] = 1
            else:
                self.bigram_counts[bigram] += 1

        for i in range(2, len(processed_data)):
            trigram = (processed_data[i-2], processed_data[i-1], processed_data[i])
            if trigram not in self.trigram_counts:
                self.trigram_counts[trigram] = 1
            else:
                self.trigram_counts[trigram] += 1

        return self.trigram_counts

    def MLE(self, alpha):
        self.probs = {}
        for trigram in self.trigram_counts:
            bigram = (trigram[0], trigram[1])
            probablity = (self.trigram_counts[trigram] + alpha) / (self.bigram_counts[bigram] + (alpha * len(self.vocab)))
            self.probs[trigram] = probablity
        
        self.default_prob = alpha / (self.bigram_counts[("<UNK>", "<UNK>")] + (alpha * len(self.vocab)))

        return self.probs

    def perplexity(self, test_data, alpha):
        self.test_data = copy.deepcopy(test_data)

        for sentence in self.test_data:
            sentence.append('<STOP>')
            sentence.insert(0, '<START>')
            sentence.insert(0, '<START>')

        self.test_words = [word for text in self.test_data for word in text]
        total_words = len(self.test_words) - 2*len(self.test_data)

        probs = self.MLE(alpha)
        log_prob = 0

        for i in range(2, len(self.test_words)):
            if self.test_words[i] not in self.vocab:
                self.test_words[i] = '<UNK>'
            if self.test_words[i-1] not in self.vocab:
                self.test_words[i-1] = '<UNK>'
            if self.test_words[i-2] not in self.vocab:
                self.test_words[i-2] = '<UNK>'
            trigram = (self.test_words[i-2], self.test_words[i-1], self.test_words[i])
            prob = probs.get(trigram, self.default_prob)
            if prob > 0:
                log_prob += math.log2(prob)

        perplexity = 2**(-log_prob / total_words)

        return perplexity

def interpolation(unigram, bigram, trigram, test_data, l1, l2, l3):
    probs = {}
    unigram_probs = unigram.MLE(0)
    bigram_probs = bigram.MLE(0)
    trigram_probs = trigram.MLE(0)

    test_data = copy.deepcopy(test_data)

    for sentence in test_data:
        sentence.append('<STOP>')
        sentence.insert(0, '<START>')
        sentence.insert(0, '<START>')


    test_words = [word for text in test_data for word in text]
    total_words = len(test_words) - 2*len(test_data)

    log_prob = 0


    for i in range(2, len(test_words)):
        trigram = (test_words[i-2], test_words[i-1], test_words[i])
        bigram = (test_words[i-1], test_words[i])
        unigram = test_words[i]
        prob = l1 * unigram_probs.get(unigram, 0) + l2 * bigram_probs.get(bigram, 0) + l3 * trigram_probs.get(trigram, 0)
        if prob > 0:
            log_prob += math.log2(prob)

    
    perplexity = 2**(-log_prob / total_words)

    return perplexity
