# Comment Sentiment Analysis
Machine learning program developed to classify social media 
comments as negative or not.


## Implementation
To use the classifier, first make a `CommentClassifier` object, train it using data, and then you may predict the classification of non-training data.  See below for an example.
```
classifier = CommentClassifier()
classifier.train(data)
classifier.predict('This is a comment to analyze!')
```


## Features
At the moment, the features of the machine learning model are only whether or not the top 1/10 of words are present.


## Data Format
Training data should be in the format of a list of tuples.  Each tuple should have a comment in the form of a string, followed by a classification that is also in the form of a string.

In the example in `project.py`, data is read from a CSV found [here](https://www.kaggle.com/fizzbuzz/cleaned-toxic-comments).  In this instance, the data set came with a variety of other classifications, such as obscene, insult, etc.  I elected to consider a comment having any type of 'toxicity' as negative.

## Dependencies
* [Natural Language Toolkit](https://www.nltk.org/)