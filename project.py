import nltk
import csv, os, pickle, random, string
# from sklearn.naive_bayes import BernoulliNB


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
        for word in self.word_features:
            features[f'contains({word})'] = (word in comment)
        return features


    def train(self, data):
        """Train the model based on data provided

        Args:
            data (list): tuples in the format (comment, classification)
        """

        all_words = []
        random.shuffle(data)
        for index, comment in enumerate(data[:10000]):
            print(index)
            tokenized_comment = nltk.word_tokenize(comment[0])
            for word in tokenized_comment:
                # stemmed_word = self.stemmer.stem(word).lower()
                stemmed_word = word.lower()
                if stemmed_word not in self.punctuation and stemmed_word not in self.stopwords:
                    all_words.append(stemmed_word)

        fd = nltk.FreqDist(word for word in all_words)
        print('Constructed frequency distribution...')
        self.word_features = list(fd)[:len(fd) // 10]
        print('Determined word features...')

        featuresets = [(self.extract_features(comment[0]), classification) for (comment, classification) in data]
        random.shuffle(featuresets)
        print('Constructed and shuffled featuresets...')
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)
        print('Trained classifier...')
        

    def predict(self, comment):
        """Predict the classification of a social media comment

        Args:
            comment (str): A comment to be classified as positive or negative
        """
        return self.classifier.classify(self.extract_features(comment))


def load_model(data):
    if (os.path.isfile('./nb_model.pkl')):
        print('Found model file...')
        with open('nb_model.pkl', 'rb') as f:
            test_classifier = pickle.load(f)
        print('Loaded model file...')
        return test_classifier

    test_classifier = CommentClassifier()
    test_classifier.train(data)
    with open('nb_model.pkl', 'wb') as f:
        pickle.dump(test_classifier, f)
    print('Dumped model to file...')
    return test_classifier
        

if __name__ == '__main__':
    # Read and classify training data
    with open('./toxiccomments/train_preprocessed.csv') as f:
        next(f)
        reader = csv.reader(f, delimiter=',')
        data = [(row[0], 'positive' if float(row[9]) == 0.0 else 'negative') for row in reader]

    test_classifier = load_model(data)

    test_comments = ('This fucking sucks!', 'This is really cool!', 'Beautiful girl!', 'Dude you are worthless', 'Shut the hell up', 'Nice!')
    for comment in test_comments:
        print(f'Comment: {comment}\t\t\tSentiment: {test_classifier.predict(comment)}')
    