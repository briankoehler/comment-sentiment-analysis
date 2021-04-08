import nltk
import random, string, time


class CommentClassifier:
    
    def __init__(self):
        """Initialize the classifier"""

        self.punctuation = set(string.punctuation)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.porter.PorterStemmer()


    def extract_features(self, comment):
        """Extracts the features of a comment

        Args:
            comment (str): Comment to extract features from
            word_features (list): Words to check for in the comment
        """

        features = {}
        comment_words = nltk.word_tokenize(comment)
        for word in self.word_features:
            features[f'contains({word})'] = (word in comment_words)
        return features


    def train(self, data):
        """Train the model based on data provided

        Args:
            data (list): Tuples in the format (comment, classification)
        """

        print('Determining all words in training data... ', end='', flush=True)
        start = time.process_time()
        all_words = []
        random.shuffle(data)
        for comment in data[:50000]:
            tokenized_comment = nltk.word_tokenize(comment[0])
            for word in tokenized_comment:
                lowercase_word = word.lower()
                if len(lowercase_word) > 1 and lowercase_word not in self.punctuation and lowercase_word not in self.stopwords:
                    all_words.append(lowercase_word)
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        print('Constructing frequency distribution... ', end='', flush=True)
        start = time.process_time()
        fd = nltk.FreqDist(word for word in all_words)
        end = time.process_time()
        print(f'Done. ({end - start}s)')
        
        print('Determining word features... ', end='', flush=True)
        start = time.process_time()
        self.word_features = list(fd)[:len(fd) // 20]
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        print('Constructing featuresets... ', end='', flush=True)
        start = time.process_time()
        featuresets = [(self.extract_features(comment), classification) for (comment, classification) in data[:50000]]
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        print('Shuffling featuresets... ', end='', flush=True)
        start = time.process_time()
        random.shuffle(featuresets)
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        print('Training classifier... ', end='', flush=True)
        start = time.process_time()
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        self.classifier.show_most_informative_features()
        

    def predict(self, comment):
        """Predict the classification of a social media comment

        Args:
            comment (str): A comment to be classified as positive or negative

        Returns:
            str: The predicted classification
        """

        return self.classifier.classify(self.extract_features(comment))


    def get_accuracy(self, test_data):
        """Returns the accuracy of the classifier on provided data set

        Args:
            test_data (list): Tuples in the format (comment, classification)

        Returns:
            float: Accuracy of the classifier
        """

        featuresets = [(self.extract_features(comment), classification) for (comment, classification) in test_data]
        return nltk.classify.accuracy(self.classifier, featuresets)