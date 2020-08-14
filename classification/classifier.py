from collections import Counter, defaultdict
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import  FastTextKeyedVectors
from gensim.models.doc2vec import Doc2Vec
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from time import time
from zeugma.embeddings import EmbeddingTransformer
import argparse 
import codecs
import os
import sys
import json 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import operator 

sys.path.append('../')
from utils import list_files_in_dir, reformat_price, calculate_label, open_json_files, plot_coefficients, FeatureExtractor


def get_label(item, topic):
    """Returns correct label based on topic and prices before and after the case"""
    if topic == 'afrekening_servicekosten':
        servicekosten_before = reformat_price(item.get('Bet.verpl. sk voor uitspraak','0'))
        servicekosten_after = reformat_price(item.get('Bet.verpl. sk na uitspraak','0'))
        label = calculate_label(servicekosten_before, servicekosten_after, topic)
    else:
        huurprijs_before = reformat_price(item.get('Huurprijs voor uitspraak', '0'))
        huurprijs_after = reformat_price(item.get('Huurprijs na uitspraak', '0'))
        if huurprijs_after == 0:
            huurprijs_after = reformat_price(item.get('Tijdelijk verlaagde huurprijs', '0'))
        label = calculate_label(huurprijs_before, huurprijs_after, topic)
    return label


def get_relevant_text_baseline(dictionary, topic):
    """Function to get the relevant text and other features from a document. The parts that 
    contain information about the decision of the case get filtered."""
    data = defaultdict(list)
    for item in dictionary:
        if 'III' not in item['text'] and 'Beoordeling' not in item['text']:
            # print("No beoordeling",item['Zaaknummer'])
            # print(item['text'])
            continue
        text = re.sub('(III)? Beoordeling.*', '', item['text'])
        text = re.sub('III Schikking.*', '', text)
        text = re.sub('III Compromis.*', '', text)
        label = get_label(item, topic)
        data['text'].append(text)
        data['labels'].append(label)
        data['applicant'].append([item['Verzoeker']])
        data['Zaaknummer'].append([item['Zaaknummer']])
    return data


def extract_features(train, test, topic):
    """Helper function to obtain dictionaries from text and topic"""
    data_train = get_relevant_text_baseline(train, topic)
    data_test = get_relevant_text_baseline(test, topic)
    return data_train, data_test


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Give arguments for running the classifier.')
    parser.add_argument('--phase', type=str, default='dev', help='Specify dev, test or all phase')
    parser.add_argument('--dates', type=bool, default=False, help='Predict the future')
    parser.add_argument('--topic', type=str, help='Give a topic to classify, all for every subject', required=True)
    parser.add_argument('--baseline', type=bool, default=False, help='Specify whether you want the baseline or not')
    return parser.parse_args()


def do_balancing(data):
    """Balances the training data so both labels have equal distribution"""
    new_data = {}
    d = [i for i,val in enumerate(data['labels']) if val =='higher' or val == 'lower']
    nd = [i for i,val in enumerate(data['labels']) if 'same' in val]
    if len(nd) < len(d):
        d = d[:len(nd)]
    elif len(nd) > len(d):
        nd = nd[:len(d)]
    new_data['text'] = [data['text'][j] for j in d] + [data['text'][i] for i in nd]
    new_data['labels'] = [data['labels'][j] for j in d] + [data['labels'][i] for i in nd]
    new_data['applicant'] = [data['applicant'][j] for j in d] + [data['applicant'][i] for i in nd]
    return new_data


def do_baselines(data_train, Xtest, phase):
    """Returns the scores of the baseline for each category"""
    clf = LinearSVC(random_state=42)
    #stop_words = stopwords.words('dutch')
    vec = CountVectorizer()
    classifier = Pipeline([('vec', vec), ('clf', clf)])
    if phase == 'dev':
        custom_cross_validate(classifier, data_train, baseline=True)
        return [], []
    classifier.fit(data_train['text'], data_train['labels'])
    return classifier, classifier.predict(Xtest)

def print_report_and_plots(classifier, dates, Xtest, ytest, yguess, topic, baseline=False, model=False):
    """Prints a classification report and plots confusion matrices and a figure with
    coefficients for each category."""
    print(topic.upper())
    print(classification_report(ytest, yguess))
    if not baseline:
        feature_names = []
        if 'ngrams' in classifier.named_steps['union'].transformer_list[0][1].named_steps:
            features_text = classifier.named_steps['union'].transformer_list[0][1].named_steps['ngrams'].get_feature_names()
            feature_names.extend(features_text)
        if model:
            feature_names.extend(200*['embedding'])
        plot_coefficients(classifier.named_steps['clf'], feature_names, topic)
        plot_confusion_matrix(classifier, Xtest, ytest, values_format='.4g')
        plt.savefig('conf_matrix_'+topic+str(dates)+'.png')

def classify_gridsearch(Xtrain, ytrain, topic):
    """"Used to gridsearch for the right parameters of the SVM"""
    clf = LinearSVC(random_state=42)
    tfidf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', clf)])
    count = Pipeline([('vec', CountVectorizer()), ('clf', clf)])
    parameters_count = {
        'vec__ngram_range': [(1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
        'vec__analyzer': ('word', 'char'),
        'clf__C': (0.1, 1, 5)
    }
    parameters_tfidf = {
        'tfidf__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],  
        'tfidf__min_df': (1, 2, 3), 
        'tfidf__norm': (None, 'l1', 'l2'),
        'tfidf__analyzer': ('word', 'char'),
        'clf__C': (0.1, 1, 5)
    }
    classifier_count = GridSearchCV(count, parameters_count, n_jobs=-1, verbose=1)
    classifier_tfidf = GridSearchCV(tfidf, parameters_tfidf, n_jobs=-1, verbose=1)
    classify_gridsearch_specific(classifier_count, Xtrain, ytrain, topic, parameters_count, CountVectorizer())
    classify_gridsearch_specific(classifier_tfidf, Xtrain, ytrain, topic, parameters_tfidf, TfidfVectorizer())

def get_best_params(clf, topic):
    """Helper function that contains the best setup for each category. Values are
    based on the results of the gridsearch."""
    stop_words = stopwords.words('dutch')
    best_params = {
        'afrekening_servicekosten': {
            'clf__C': 5,
            'tfidf__analyzer': 'word', 
            'tfidf__min_df': 1,
            'tfidf__ngram_range': (5, 5),
            'tfidf__norm': 'l2',
            'tfidf__lowercase': False,
            'tfidf__stop_words': stop_words,
        },
        'gebreken': {
            'clf__C': 1,
            'vec__analyzer': 'word',
            'vec__ngram_range': (5, 5),
            'vec__lowercase': False,
            'vec__stop_words': stop_words,
        },
        'melding': {
            'clf__C': 0.1, 
            'vec__analyzer': 'word', 
            'vec__ngram_range': (3, 4),
            'vec__lowercase': True,
            'vec__stop_words': stop_words,
        },
        'punten': {
            'clf__C': 5, 
            'tfidf__analyzer': 'word',
            'tfidf__min_df': 2,
            'tfidf__ngram_range': (1, 4),
            'tfidf__norm': 'l2',
            'tfidf__lowercase': True,
            'tfidf__stop_words': None,
        },
        'toetsing': {
            'clf__C': 1,
            'tfidf__analyzer': 'word', 
            'tfidf__min_df': 3,
            'tfidf__ngram_range': (1, 5),
            'tfidf__norm': 'l2',
            'tfidf__lowercase': True,
            'tfidf__stop_words': stop_words,
        },
        'woningverbetering': {
            'clf__C': 0.1,
            'vec__analyzer': 'char',
            'vec__ngram_range': (5, 5),
            'vec__lowercase': True,
            'vec__stop_words': stop_words,
        },
    }
    if topic in ['gebreken', 'melding', 'woningverbetering']:
        classifier = Pipeline([('vec', CountVectorizer()), ('clf', clf)])
    else:
        classifier = Pipeline([('tfidf', TfidfVectorizer()), ('clf', clf)])
    classifier.set_params(**best_params[topic])
    return classifier


class FindFastTextEmbeddings(TransformerMixin, BaseEstimator):
    """Class to extract FastText Embeddings """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return self.model[X]


class ItemSelector(BaseEstimator, TransformerMixin):
    """Custom class to get the right features. 
    self.key is the name of a feature"""
    def __init__(self, key):
        self.key = key 
    
    def fit(self, x, y=None):
        return self 
    
    def transform(self, data_dict):
        return data_dict[self.key]

class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    """Class that transforms a document into a doc2vec representation"""
    def __init__(self, model):
        self.model = model 

    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        new_Xtrain = []
        for item in data_dict:
            doc = []
            for sent in sent_tokenize(item):
                doc.extend(word_tokenize(sent))
            vector = self.model.infer_vector(doc)
            new_Xtrain.append(vector)
        return new_Xtrain

def get_applicant(clf):
    """Uses one hot encoding to convert the applicant to numbers"""
    applicants = OneHotEncoder()
    return Pipeline([('applicants', applicants), ('clf', clf)])


def get_fasttext_embeddings(model, clf):
    """Helper function to get fasttext embeddings"""
    embeddings = FindFastTextEmbeddings(model=model)
    classifier = Pipeline([('embeddings', embeddings), ('clf', clf)])
    return classifier


def get_word2vec_embeddings(model, clf):
    """Helper function to get word2vec embeddings"""
    embeddings = EmbeddingTransformer(model=model)
    classifier = Pipeline([('embeddings', embeddings), ('clf', clf)])
    return classifier


def get_doc2vec_embeddings(model, clf):
    """Helper function to get doc2vec embeddings"""
    embeddings = Doc2VecTransformer(model=model)
    classifier = Pipeline([('embeddings', embeddings), ('clf', clf)])
    return classifier
    
def classify_gridsearch_specific(classifier, Xtrain, ytrain, topic, parameters, vec):
    """Function that runs the gridsearch and displays the best models and scores."""
    t0 = time()
    classifier.fit(Xtrain, ytrain)
    print(topic.upper())
    print(f"Done in {time() -t0}s")
    print(f"Best score: {classifier.best_score_}")
    print("Best parameters set:")             
    best_parameters = classifier.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
    pipeline = Pipeline([('vec', vec), ('clf', LinearSVC(random_state=42))])
    pipeline.set_params(**best_parameters)   
    classify(pipeline, Xtrain, ytrain, topic)


def combine_embeddings_ngrams(w2v_model, d2v_model, clf, topic):
    """Helper function that combines the word embeddings, document
    embeddings and n-grams."""
    return Pipeline([
        ('extract_features', FeatureExtractor()),
        ('union', FeatureUnion(
            transformer_list = [
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('ngrams', get_best_params(clf, topic)[0]),
                ])),
                ('word_embeddings', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('embeddings', get_word2vec_embeddings(w2v_model, clf)[0]),
                ])),
                ('doc_embeddings', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('docs', get_doc2vec_embeddings(d2v_model, clf)[0]),
                ])),
            ],
        )),
        ('clf', get_best_params(clf, topic)[1])
    ])

def custom_cross_validate(pipeline, data, cv=5, baseline=False):
    """Own implementation of cross_validation for when the standard, better optimized
    version of sklearn would not work."""
    kf = StratifiedKFold(n_splits=cv, shuffle=False)
    scores, predictions, real = [], [], []
    keys = data.keys()
    for train_index, test_index in kf.split(data['text'], data['labels']):
        new_data_train = {}
        new_data_test = {}
        for key in keys:
            new_data_train[key] = np.array(data[key])[train_index]
            new_data_test[key] = np.array(data[key])[test_index]
        train_labels = new_data_train['labels']
        test_labels = new_data_test['labels']
        if baseline:
            new_data_train = new_data_train['text']
            new_data_test = new_data_test['text']
        model = pipeline.fit(new_data_train, train_labels)
        predictions.extend(pipeline.predict(new_data_test))
        real.extend(test_labels)
        scores.append(model.score(new_data_test, test_labels))
    print('scores from each iteration:', scores)
    print('Average K-Fold Score: ', np.mean(scores))
    print('Standard deviation:', np.std(scores))
    return predictions, real
    
def classify(pipeline, Xtrain, ytrain, topic):
    """Fits a pipeline, outputs scores and shows reports"""
    pipeline.fit(Xtrain, ytrain)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(cross_val_score(pipeline, Xtrain, ytrain, cv=cv))
    yguess = cross_val_predict(pipeline, Xtrain ,ytrain, cv=cv)
    print_report_and_plots(pipeline, ytrain, yguess, topic)


def classify_dev_combined(pipeline, data, topic):
    """Fits a pipeline, outputs scores and shows reports when normal cross
    validation does not work."""
    pipeline.fit(data, data['labels'])
    yguess, ytest = custom_cross_validate(pipeline, data)
    print_report_and_plots(pipeline, data, ytest, yguess, topic)

def final_experiment(clf, data_train, data_test, topic, phase, w2v_model=False):
    """Function that uses the best model to test the test sets on."""
    if topic == 'afrekening_servicekosten':
        pipeline = Pipeline([
            ('extract_features', FeatureExtractor()),
            ('union', FeatureUnion(
                transformer_list = [
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('ngrams', get_best_params(clf, topic)[0]),
                    ])),
                    ('word_embeddings', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('embeddings', get_word2vec_embeddings(w2v_model, clf)[0]),
                    ])),
                ],
            )),
            ('clf', get_best_params(clf, topic)[1])
        ])
    elif topic == 'woningverbetering':
        pipeline = Pipeline([
            ('extract_features', FeatureExtractor()),
            ('union', FeatureUnion(
                transformer_list = [
                    ('word_embeddings', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('embeddings', get_word2vec_embeddings(w2v_model, clf)[0]),
                    ])),
                ],
            )),
            ('clf', get_best_params(clf, topic)[1])
        ])
    else:
        pipeline = Pipeline([
            ('extract_features', FeatureExtractor()),
            ('union', FeatureUnion(
                transformer_list = [
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('ngrams', get_best_params(clf, topic)[0]),
                    ])),
                ],
            )),
            ('clf', get_best_params(clf, topic)[1])
        ])
    pipeline.fit(data_train, data_train['labels'])
    yguess = pipeline.predict(data_test)
    print_report_and_plots(pipeline, data_train['text'], data_test, data_test['labels'], yguess, topic, model=w2v_model)
    if topic == 'melding':
        yguess = yguess.tolist()
        case_numbers = []
        print(len(yguess))
        print(len(data_test['Zaaknummer']))
        for i, j in enumerate(yguess):
            if j != data_test['labels'][i]:
                case_numbers.append(data_test['Zaaknummer'][i])
        print(case_numbers)

def do_heuristic(ytrain, ytest):
    """Outputs most frequent outcome in the past"""
    most_frequent = Counter(ytrain).most_common()[0][0]
    yguess = [most_frequent] *len(ytest)
    print("Heuristic accuracy: ", accuracy_score(yguess, ytest))

def main():
    """Main function that adds everything together. Some function calls are commented out when they were
    not used the last time the program was run."""
    args = parse_args()
    if args.topic == 'all':
        topics = ['afrekening_servicekosten', 'gebreken', 'melding', 'punten', 'toetsing', 'woningverbetering']
    if args.dates == True:
        topics = ['afrekening_servicekosten', 'gebreken', 'punten', 'toetsing']
    if 'topics' not in locals():
        topics = [args.topic]
    
    #fasttext = FastTextKeyedVectors.load('../embeddings/fasttext_200d25e.txt')
    #doc200 = Doc2Vec.load("../embeddings/doc2vec_200.txt")
    cbow200 = KeyedVectors.load_word2vec_format("../embeddings/custom_w2v_200d_cbow.txt", binary=False)
    skip200 = KeyedVectors.load_word2vec_format("../embeddings/custom_w2v_200d_skipgram.txt", binary=False)
    for topic in topics:
        if args.phase == 'all':
            train_json = open_json_files(topic, False, 'train')
            old_test = open_json_files(topic, False, 'test')
            train_json = train_json + old_test
            test_json = open_json_files(topic, True, 'test')
        else:
            train_json = open_json_files(topic, args.dates, 'train')
            test_json = open_json_files(topic, args.dates, 'test')
        data_train, data_test = extract_features(train_json, test_json, topic)
        label_distribution = data_train['labels']
        data_train = do_balancing(data_train)
        Xtrain = data_train['text']
        ytrain = data_train['labels']
        Xtest = data_test['text']
        ytest = data_test['labels']
        if args.baseline:   
            print(topic.upper())     
            classifier, yguess = do_baselines(data_train, Xtest, args.phase)
            if args.phase == 'test':
                print_report_and_plots(classifier, args.dates, Xtest, ytest, yguess, topic, args.dates, baseline=True)
        clf = LinearSVC(random_state=42)
        if args.phase != 'dev':
            phase = args.phase + str(args.dates)
            if topic == 'afrekening_servicekosten':
                final_experiment(clf, data_train, data_test, topic, phase, skip200)
            elif topic == 'woningverbetering': 
                final_experiment(clf, data_train, data_test, topic, phase, cbow200)
            else:
                final_experiment(clf, data_train, data_test, topic, phase)
            if args.dates == True:
                do_heuristic(label_distribution, ytest)
                
        else:
            print(topic.upper())
            if topic in ['melding', 'punten', 'woningverbetering']:
                pipeline = combine_embeddings_ngrams(cbow200, doc300, clf, topic)
            elif topic in ['gebreken, toetsing']:
                pipeline = combine_embeddings_ngrams(cbow300, doc300, clf, topic)
            else:
                pipeline = combine_embeddings_ngrams(skip200, doc200, clf, topic)
            classify_dev_combined(pipeline, data_train, topic)
            
               
if __name__ == "__main__":
    main()
