import nltk
import random, string, time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Done. ({time.time() - start}s)')
        return result
    return wrapper


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

    
    @timer
    def determine_words(self, data):
        all_words = []

        for comment in data[:50000]:
            tokenized_comment = nltk.word_tokenize(comment[0])
            for word in tokenized_comment:
                lowercase_word = word.lower()
                if len(lowercase_word) > 1 and lowercase_word not in self.punctuation and lowercase_word not in self.stopwords:
                    all_words.append(lowercase_word)

        return all_words

    
    @timer
    def construct_fd(self, all_words):
        return nltk.FreqDist(word for word in all_words)


    @timer
    def determine_word_features(self, fd):
        return list(fd)[:len(fd) // 20]


    @timer
    def construct_featuresets(self, data):
        return [(self.extract_features(comment), classification) for (comment, classification) in data[:50000]]


    @timer
    def train_classifier(self, featuresets):
         return nltk.NaiveBayesClassifier.train(featuresets)


    def train(self, data):
        """Train the model based on data provided

        Args:
            data (list): Tuples in the format (comment, classification)
        """

        random.shuffle(data)

        print('Determining all words in training data... ', end='', flush=True)
        all_words = self.determine_words(data)

        print('Constructing frequency distribution... ', end='', flush=True)
        fd = self.construct_fd(all_words)
        
        print('Determining word features... ', end='', flush=True)
        self.word_features = self.determine_word_features(fd)

        print('Constructing featuresets... ', end='', flush=True)
        featuresets = self.construct_featuresets(data)

        random.shuffle(featuresets)

        print('Training classifier... ', end='', flush=True)
        self.classifier = self.train_classifier(featuresets)

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