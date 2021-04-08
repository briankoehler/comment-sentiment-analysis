import argparse, csv, os, pickle, time
from classifier import CommentClassifier
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train_load')
    parser.add_argument('-c', '--classify', nargs=1)
    args = parser.parse_args()

    if args.train_load == 'train':
        # Read and classify training data
        print('Reading and classifying training data... ', end='', flush=True)
        start = time.process_time()
        with open('./data/sentiment140/trainingdata.csv', encoding='ISO-8859-1') as f:
            reader = csv.reader(f, delimiter=',')
            training_data = [(row[5], 'not-negative' if row[0] in ('2', '4') else 'negative') for row in reader]
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        # Create model
        classifier = CommentClassifier()

        # Train model
        classifier.train(training_data)

        # Check for previous model
        if os.path.isfile('./models/model.pkl'):
            print('Renaming old model... ', end='', flush=True)
            start = time.process_time()
            os.rename('./models/model.pkl', './models/old_model.pkl')
            end = time.process_time()
            print(f'Done. ({end - start}s)')

        # Save model
        print('Dumping model to file... ', end='', flush=True)
        start = time.process_time()
        with open('./models/model.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        end = time.process_time()
        print(f'Done. ({end - start}s)')

    elif args.train_load == 'load':
        # Locate model
        if os.path.isfile('./models/model.pkl'):
            print('Loading model file... ', end='', flush=True)
            start = time.process_time()
            with open('./models/model.pkl', 'rb') as f:
                classifier = pickle.load(f)
            end = time.process_time()
            print(f'Done. ({end - start}s)')

        else:
            print('No model file found...')
            quit()

    if not args.classify:
        # Read and classify test data
        print('Reading and classifying test data... ', end='', flush=True)
        start = time.process_time()
        with open('./data/sentiment140/testdata.csv', encoding='ISO-8859-1') as f:
            reader = csv.reader(f, delimiter=',')
            test_data = [(row[5], 'not-negative' if row[0] in ('2', '4') else 'negative') for row in reader]
        end = time.process_time()
        print(f'Done. ({end - start}s)')

        # # Get accuracy
        print(f'Accuracy: {classifier.get_accuracy(test_data)}')

    else:
        print(f'Prediction: {classifier.predict(args.classify[0])}')
    