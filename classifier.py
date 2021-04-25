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


    def extract_features(self, comment):
        """Extracts the features of a comment

        Args:
            comment (str): Comment to extract features from
            word_features (list): Words to check for in the comment
        """

        # Initialize features
        features = {}

        # Tokenize, converting to lower case, removing characters repeating more than 3x, and stripping handles
        comment_words = nltk.word_tokenize(comment)
        lowecase_comment_words = [token.lower() for token in comment_words]

        # Determine if features present
        for word in self.word_features:
            features[f'contains({word})'] = (word in lowecase_comment_words)

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
        for comment in data[:50000]:

            # Tokenize, converting to lower case, removing characters repeating more than 3x, and stripping handles
            tokenized_comment = nltk.word_tokenize(comment[0])

            # Adding words to list if 3+ characters and not a stopword
            for word in tokenized_comment:
                lowercase_word = word.lower()
                if len(lowercase_word) > 2 and lowercase_word not in self.punctuation and lowercase_word not in self.stopwords:
                    all_words.append(word)

        return all_words

    
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


    @timer
    def construct_featuresets(self, data: list) -> list:
        """Constructs featuresets using the provided data

        Args:
            data (list): Tuples in the format (comment, classification)

        Returns:
            list: Featuresets of training data
        """

        return [(self.extract_features(comment), classification) for (comment, classification) in data[:50000]]


    @timer
    def train_classifier(self, featuresets: list) -> nltk.NaiveBayesClassifier:
        """Trains the Naive Bayes Classifier using the featuresets.

        Args:
            featuresets (list): Featuresets of training data

        Returns:
            NaiveBayesClassifier: Trained model
        """

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
