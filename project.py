import argparse, csv, os, pickle
from classifier import CommentClassifier
from time_track import timer


def parse_args() -> argparse.Namespace:
    """Parses arguments passed via the terminal.

    Returns:
        argsparse.Namespace: Parsed arguments
    """

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('train_load')
    parser.add_argument('-c', '--classify', nargs=1)

    # Parse and return
    args = parser.parse_args()
    return args


@timer
def preprocess_data(filepath: str, encoding: str = 'ISO-8859-1') -> list:
    """Read a data file and classify each point.  Expecting the CSV format to be the same as 
    Sentiment140's.  Each point will be either negative, or not-negative.

    Args:
        filepath (str): Path to the data CSV file
        encoding (str, optional): Encoding to read the file. Defaults to 'ISO-8859-1'.

    Returns:
        list: Pre-processed data
    """

    with open(filepath, encoding=encoding) as f:
        reader = csv.reader(f, delimiter=',')
        return [(row[5], 'not-negative' if row[0] in ('2', '4') else 'negative') for row in reader]


@timer
def rename_old_model(model_path: str, new_name: str):
    """Rename the old model file.  Typically used to make space for the new model file.

    Args:
        model_path (str): Filepath of most recent model
        new_name (str): New filepath to assign
    """

    os.rename(model_path, new_name)


@timer
def dump_model(model_path: str, classifier: CommentClassifier):
    """Dump a CommentClassifier instance to a file using pickle.

    Args:
        model_path (str): Filepath to produce model file
        classifier (CommentClassifier): Model to cache
    """

    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)


@timer
def load_model(model_path: str) -> CommentClassifier:
    """Load a CommentClassifier from a pickle file

    Args:
        model_path (str): Filepath of model to load

    Returns:
        CommentClassifier: Model from file
    """

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def train_model(filepath: str, save: bool = True, model_path: str = './models/model.pkl', old_model: str = './models/old_model.pkl') -> CommentClassifier:
    """Train a model from the CSV located at filepath.

    Args:
        filepath (str): Filepath of data
        save (bool, optional): Whether or not to save the model to a filef for future use. Defaults to True.
        model_path (str, optional): Filepath to pickle model file. Defaults to './models/model.pkl'.
        old_model (str, optional): Filepath of previous model. Defaults to './models/old_model.pkl'.

    Returns:
        CommentClassifier: Trained classifier
    """

    print('Reading and classifying training data... ', end='', flush=True)
    training_data = preprocess_data(filepath)

    # Create model
    classifier = CommentClassifier()

    # Train model
    classifier.train(training_data)

    if save:
        # Check for previous model
        if os.path.isfile(model_path):
            print('Renaming old model... ', end='', flush=True)
            rename_old_model(model_path, old_model)

        # Save model
        print('Dumping model to file... ', end='', flush=True)
        dump_model(model_path, classifier)

    return classifier
        

if __name__ == '__main__':

    args = parse_args()

    ##################
    # Train vs. Load #
    ##################
    if args.train_load == 'train':
        classifier = train_model('./data/sentiment140/trainingdata.csv')

    elif args.train_load == 'load':
        # Locate model
        if os.path.isfile(model_path):
            print('Loading model file... ', end='', flush=True)
            classifier = load_model(model_path)

        else:
            print('No model file found...')
            quit()

    ##############################
    # Classify vs. Test Accuracy #
    ##############################
    if not args.classify:
        # Read and classify test data
        print('Reading and classifying test data... ', end='', flush=True)
        test_data = preprocess_data('./data/sentiment140/testdata.csv')

        # # Get accuracy
        print(f'Accuracy: {classifier.get_accuracy(test_data)}')
        quit()

    print(f'Prediction: {classifier.predict(args.classify[0])}')
    