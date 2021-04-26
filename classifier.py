import nltk
import random, string
from time_track import timer


class CommentClassifier:
    
    def __init__(self):
        """Initialize the classifier"""

        self.punctuation = set(string.punctuation)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.training_size = 50000


    def extract_features(self, comment: str) -> dict:
        """Extracts the features of a comment

        Args:
            comment (str): Comment to extract features from
            word_features (list): Words to check for in the comment
        """

        # Initialize features
        features = {}

        # Tokenize, converting to lower case, removing characters repeating more than 3x, and stripping handles
        comment_words = [token.lower() for token in nltk.word_tokenize(comment)]

        # Determine if features present
        for word in self.word_features:
            features[f'contains({word})'] = (word in comment_words)

        return features

    
    @timer
    def determine_words(self, data: list) -> list:
        """Produces list of all words (including repeats)

        Args:
            data (list): Tuples in the format (comment, classification)

        Returns:
            list: List of all words (including repeats)
        """

        # Initialize list of words
        all_words = []

        # Take first 50,0000 data points (assume pre-shuffled)
        for comment in data[:self.training_size]:

            # Tokenize, converting to lower case, removing characters repeating more than 3x, and stripping handles
            tokenized_comment = nltk.word_tokenize(comment[0])

            # Adding words to list if 3+ characters and not a stopword
            for word in tokenized_comment:
                lowercase_word = word.lower()
                if len(lowercase_word) > 2 and lowercase_word not in self.punctuation and lowercase_word not in self.stopwords:
                    all_words.append(word)

        return all_words


    @timer
    def determine_bigrams(self, data: list) -> list:
        all_bigrams = []

        for comment in data[:self.training_size]:
            tokenized_comment = nltk.casual_tokenize(comment[0], preserve_case=False, reduce_len=True, strip_handles=True)
            filtered_comment = [token for token in tokenized_comment if len(token) > 2 and token not in self.punctuation and token not in self.stopwords]
            all_bigrams += list(nltk.bigrams(filtered_comment))

        return all_bigrams

    
    @timer
    def construct_fd(self, all_words: list) -> nltk.FreqDist:
        """Constructs a FD from a given list of words

        Args:
            all_words (list): List of all words in training set

        Returns:
            nltk.FreqDist: FD of words in training set
        """

        return nltk.FreqDist(word for word in all_words)


    @timer
    def determine_word_features(self, fd: nltk.FreqDist) -> list:
        """Determines the top %5 of words into a list

        Args:
            fd (nltk.FreqDist): FD of all words in a training set

        Returns:
            list: Top 5% of words
        """

        return list(fd)[:len(fd) // 20]
        # return list(fd)[:1000]


    @timer
    def construct_featuresets(self, data: list) -> list:
        """Constructs featuresets using the provided data

        Args:
            data (list): Tuples in the format (comment, classification)

        Returns:
            list: Featuresets of training data
        """

        return [(self.extract_features(comment), classification) for (comment, classification) in data[:self.training_size]]


    @timer
    def train_classifier(self, featuresets: list) -> nltk.NaiveBayesClassifier:
        """Trains the Naive Bayes Classifier using the featuresets.

        Args:
            featuresets (list): Featuresets of training data

        Returns:
            NaiveBayesClassifier: Trained model
        """

        # return nltk.NaiveBayesClassifier.train(featuresets)
        return nltk.NaiveBayesClassifier.train(featuresets)


    def train(self, data: list):
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
        

    def predict(self, comment: str) -> str:
        """Predict the classification of a social media comment

        Args:
            comment (str): A comment to be classified as positive or negative

        Returns:
            str: The predicted classification
        """

        return self.classifier.classify(self.extract_features(comment))


    def get_accuracy(self, test_data: list) -> float:
        """Returns the accuracy of the classifier on provided data set

        Args:
            test_data (list): Tuples in the format (comment, classification)

        Returns:
            float: Accuracy of the classifier
        """

        featuresets = [(self.extract_features(comment), classification) for (comment, classification) in test_data]
        return nltk.classify.accuracy(self.classifier, featuresets)
