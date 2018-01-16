#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from unidecode import unidecode
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords

import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import model_selection
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

SEED=7
KFOLDS=5
SCORING='accuracy'
TRAINING_SET="data/training.csv"

class Classifier:
    def __init__(self):        
        self.vectorizer = CountVectorizer(ngram_range=(1,2))        
        self.classifier = MultinomialNB()

    # Read traning/validation data and return labels and text
    def read_data(self, path):
        labels = []
        inputs = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                labels.append(row[0])
                inputs.append(row[1])
        return (labels, inputs)

    # Remove stop words, accents, uppercase and stem
    def preprocess_text(self, text):
        tokens = []
        stemmer = RSLPStemmer()
        for t in text.split():
            # Need a better set of stopwords
            #if t in stopwords.words('portuguese'):
                #continue
            t = unidecode(t)
            t = t.lower()
            t = re.sub(r'\W+', '', t)
            t = stemmer.stem(t)
            tokens.append(t)
        return ' '.join(tokens)

    # Bake a feature vector with given data
    def bake_data(self, path):
        labels, text = self.read_data(path)
        data = map(self.preprocess_text, text)
        matrix = self.vectorizer.fit_transform(data).todense()
        return labels, matrix

    # Load and train a classifier
    def load_classifier(self, training_data_path):        
        Y_train, X_train = self.bake_data(training_data_path)
        self.classifier.fit(X_train, Y_train)

    # Classify
    def classify(self, text):
        text = self.preprocess_text(text)
        array = self.vectorizer.transform([text]).todense()
        # Get each class probabilities
        probabilities = self.classifier.predict_proba(array)[0]
        # Get the predicted class
        result = self.classifier.predict(array)[0]
        # This is an empirical observation for bad classifications
        test_probabilities = np.sort(probabilities)[:-3:-1]
        if test_probabilities[0] < 0.5 and np.sum(test_probabilities) > 0.5:
            result  = 'U'
        # Return the class and sorted probabilities
        return (result, np.sort(probabilities))

    def test_models(self, training_data_path):
        # Models to test
        models = []
        models.append(('LR',   LogisticRegression()))
        models.append(('PER',  Perceptron(max_iter=1000, tol=1e-3)))
        models.append(('KNN',  KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('MNB',  MultinomialNB()))
        models.append(('GNB',  GaussianNB()))
        models.append(('SVM',  SVC()))
        models.append(('SGD',  SGDClassifier(max_iter=1000, tol=1e-3)))

        # Run the classifier with each model
        names  = []
        scores = []
        for name, model in models:
            # Bake the data
            Y_train, X_train = self.bake_data(training_data_path)
            # K-fold the data
            kfolds = model_selection.KFold(n_splits=KFOLDS, random_state=SEED)
            # Do a cross validation with the k-folded data
            cv     = model_selection.cross_val_score(model, X_train, Y_train,
                                                      cv=kfolds, scoring=SCORING)
            # Show the results
            names.append(name)
            scores.append(cv)
            print('{}: {} ({})'.format(name, cv.mean(), cv.std()))

def main():
    # A few tests and examples
    c = Classifier()
    # Test different models to select the best
    c.test_models(TRAINING_SET)
    # Then test the classifier with a selected model from above
    c.load_classifier(TRAINING_SET)
    print(c.classify("olá, bom dia"))
    print(c.classify("pode me ajudar?"))
    print(c.classify("gostaria de saber onde está o meu pedido"))
    print(c.classify("onde está meu pedido?"))
    print(c.classify("minhas compras não chegam!"))
    print(c.classify("quero minha encomendaaa!"))
    print(c.classify("oi!"))
    print(c.classify("e aí?"))
    print(c.classify("amanhã chove"))
    print(c.classify("até mais"))
    print(c.classify("tchau"))

if __name__ == "__main__":
    main()

