from sklearn.linear_model import LinearRegression 

from collections import Counter, defaultdict
from datetime import datetime
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import  FastTextKeyedVectors
from gensim.models.doc2vec import Doc2Vec

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from scipy.stats import shapiro, boxcox, levene, spearmanr
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDRegressor, Lasso, ElasticNet, RANSACRegressor, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score, KFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.utils._testing import ignore_warnings
from time import time
from zeugma.embeddings import EmbeddingTransformer
import argparse 
import codecs
import os
import sys
import json 
import math
import re
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels as sm
import itertools 

sys.path.append('../')
from utils import list_files_in_dir, reformat_price, open_json_files, get_topic, plot_coefficients, FeatureExtractor


def get_price_after(item, topic):
    """Get the price after a case, in case this is 0, get the temporary reducted price"""
    if topic == 'afrekening_servicekosten':
        servicekosten_after = reformat_price(item.get('Bet.verpl. sk na uitspraak','0'))
        label = servicekosten_after
    else:
        huurprijs_after = reformat_price(item.get('Huurprijs na uitspraak', '0'))
        if huurprijs_after == 0:
            huurprijs_after == reformat_price(item.get('Tijdelijk verlaagde huurprijs', '0'))
        label = huurprijs_after
    return label


def get_price_before(item, topic):
    """Get the price before the case"""
    if topic == 'afrekening_servicekosten':
        servicekosten_before = reformat_price(item.get('Bet.verpl. sk voor uitspraak','0'))
        label = servicekosten_before
    else:
        huurprijs_before = reformat_price(item.get('Huurprijs voor uitspraak', '0'))
        label = huurprijs_before
    return label

def calc_label(label, price_before, price_after):
    """Returns the price, difference between old and new price or the percentual difference"""
    if label == 'dif':
        return abs(price_after-price_before)
    elif label == 'per':
        return (abs(price_after - price_before) / price_before) * 100
    else:
        return price_after


def find_statement_tenant(text):
    """Returns the statement of a tenant. Returns None if no statement is found """
    statement_tenant = re.search(
        r'(Korte samenvatting .+?\shuurder:?)(.*)(Korte)?', text)
    if statement_tenant:
        statement_tenant = "".join(
        [ch for ch in statement_tenant.group(2) if ord(ch)<128])
        return statement_tenant.split()
    return ''

def find_statement_landlord(text):
    """Returns the statement of a landlord. Returns None if no statement is found"""
    statement_landlord = re.search(
        r'(Korte samenvatting .+?verhuurder:?)(.*)', text)
    if statement_landlord:
        statement_landlord = "".join(
            [ch for ch in statement_landlord.group(2) if ord(ch)<128])
        return statement_landlord.split()
    return ''

def get_relevant_text_baseline(dictionary, topic, label):
    """Function to get the relevant text and other features from a document. The parts that 
    contain information about the decision of the case get filtered and some 
    outliers are corrected."""
    data = defaultdict(list)
    for item in dictionary:
        if 'III' not in item['text'] and 'Beoordeling' not in item['text']:
            # print("No beoordeling",item['Zaaknummer'])
            # print(item['text'])
            continue
        text = re.sub('(III)? Beoordeling.*', '', item['text'])
        text = re.sub('III Schikking.*', '', text)
        text = re.sub('III Compromis.*', '', text)
        price_before = get_price_before(item, topic)
        price_after = get_price_after(item, topic)
        if item['Zaaknummer'] == 'ZKN-2015-008837':
            price_after = price_before
        if item['Zaaknummer'] == 'ZKN-2016-007636':
            price_before = 720.71
        if item['Zaaknummer'] == 'ZKN-2015-001326':
            price_before = 302.04
        if item['Zaaknummer'] == 'ZKN-2016-001436':
            price_before = 205.00
        if item['Zaaknummer'] == 'ZKN-2017-003158':
            price_after = 207.5
        if item['Zaaknummer'] == 'ZKN-2016-008130':
            price_before = 160.46
        # if price_before or price_after > 900:
        #     print(item['Zaaknummer'])
        if price_before < 2:
            continue
        if price_after == 0:
            price_after = 0.01
        if price_after != price_before:
            data['text'].append(text)            
            data['labels'].append(calc_label(label, price_before, price_after))
            data['price'].append(float(price_before))
            data['applicant'].append(item['Verzoeker'])
            data['date'].append(datetime.strptime(item['Datum afdoening'], '%d-%m-%y'))
            regex = re.compile(r'zijn\sniet\ster\szitting\sverschenen')
            if 'Beide partijen hebben niet' in text or regex.search(text.lower()) or 'geen der partijen' in text.lower():
                data['lentenant'].append(0)
                data['lenlandlord'].append(0)
            else:
                data['lentenant'].append(len(find_statement_tenant(text)))
                data['lenlandlord'].append(len(find_statement_landlord(text)))
                if len(find_statement_landlord(text)) > 10000:
                    print(text, item['Zaaknummer'])
            data['city'].append(item['Plaats'])
            typewoonruimte = item.get('Type woonruimte', 'Unknown')
            if typewoonruimte == 'Onvrij':
                typewoonruimte = 'Onzelfstandig'
            elif typewoonruimte == 'Woonwagen of woonwagenstandplaats':
                typewoonruimte = 'Woonwagen'
            data['typewoonruimte'].append(typewoonruimte)
    return pd.DataFrame.from_dict(data)


def extract_features(train, test, topic, label):
    """helper function to obtain pandas dataframes from text, topic and label"""
    data_train = get_relevant_text_baseline(train, topic, label)
    data_test = get_relevant_text_baseline(test, topic, label)
    return data_train, data_test


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Give arguments for running the classifier.')
    parser.add_argument('--phase', type=str, default='dev', help='Specify dev or test phase')
    parser.add_argument('--dates', type=bool, default=False, help='Predict the future')
    parser.add_argument('--topic', type=str, help='Give a topic to classify, all for every subject', required=True)
    parser.add_argument('--baseline', type=bool, default=False, help='Specify whether you want the baseline or not')
    parser.add_argument('--label', type=str, default='abs', help='Specify the target, abs, dif or per')
    return parser.parse_args()

class ItemSelector(BaseEstimator, TransformerMixin):
    """Custom class to get the right features. 
    self.key is the name of a feature"""
    def __init__(self, key):
        self.key = key 
    
    def fit(self, x, y=None):
        return self 
    
    def transform(self, data_dict):
        return data_dict[self.key]
    
    def get_feature_names(self, data_dict):
        return data_dict[self.key]


class CustomTransformer(BaseEstimator, TransformerMixin):
    """Transforms a list into a numpy array and makes it compatible
    by reshaping."""
    def fit(self, x, y=None):
        return self 

    def transform(self, data_dict):
        return np.array([x for x in data_dict]).reshape(-1, 1)

def get_word2vec_embeddings(model, clf):
    """Helper function to get word2vec embeddings"""
    embeddings = EmbeddingTransformer(model=model)
    classifier = Pipeline([('embeddings', embeddings), ('clf', clf)])
    return classifier

class DummyEstimator(BaseEstimator):
    """Dummy Estimator to act as a placeholder."""
    def fit(self): pass
    def score(self): pass

def gridsearch_regression(data_train, topic, label, models=False):
    """Gridsearches for the right model if models=True or searches for the best N-grams setup
    if models=False"""
    t0 = time()
    if models:
        vec = CountVectorizer()
        pipeline = Pipeline([
            ('extract_features', FeatureExtractor()),
            ('union', FeatureUnion(
                transformer_list=[
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='text')),
                        ('vec', vec),
                    ])),
                    ('price', Pipeline([
                        ('selector', ItemSelector(key='price')),
                        ('customtrans', CustomTransformer()),
                        #('scaler', StandardScaler()),
                    ])),
                    ('city', Pipeline([
                        ('selector', ItemSelector(key='city')),
                        ('customtrans', CustomTransformer()),
                        ('cityt', OneHotEncoder(handle_unknown='ignore')),
                    ])),
                    ('applicant', Pipeline([
                        ('selector', ItemSelector(key='applicant')),
                        ('customtrans', CustomTransformer()),
                        ('appl', OneHotEncoder()),
                    ])),
                    ('typewoonruimte', Pipeline([
                        ('selector', ItemSelector(key='typewoonruimte')),
                        ('customtrans', CustomTransformer()),
                        ('typew', OneHotEncoder(handle_unknown='ignore')),
                    ])),
                ],
            )),
            ('clf',  DummyEstimator()) #placeholder
        ])
        search_space = {'clf': [LinearRegression(), ElasticNet(), Lasso(), Ridge(), LinearSVR()]}
        
    if not models and topic != 'afrekening_servicekosten':
        vec = CountVectorizer()
        if topic in ['afrekening_servicekosten', 'gebreken']:
            pipeline = Pipeline([('vec', vec), ('clf', Lasso(max_iter=10000))])
            search_space = {
            'vec__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
            'vec__min_df': (1,2,3),
            'vec__analyzer': ('word', 'char'),
            'clf__normalize': (True, False),
            'clf__selection': ('cyclic', 'random'), 
            }
        elif topic == 'woningverbetering':
            pipeline = Pipeline([('vec', vec), ('clf', LinearRegression())])
            search_space = {
                'vec__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
                'vec__min_df': (1,2,3),
                'vec__analyzer': ('word', 'char'),
                'clf__normalize': (True, False),
            }
        else:
            #Melding, punten, toetsing
            pipeline = Pipeline([('vec', vec), ('clf', ElasticNet(max_iter=10000))])
            search_space = {
            'vec__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
            'vec__min_df': (1,2,3),
            'vec__analyzer': ('word', 'char'),
            'clf__normalize': (True, False),
            'clf__selection': ('cyclic', 'random'),
        }
    
        
        
        gs = GridSearchCV(pipeline, search_space, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=1)
        labels = data_train['labels']
        if models:
            X = data_train.drop('labels', axis=1)
        else:
            X = data_train['text']
        gs.fit(X, labels)
        print(topic.upper())
        print(f"COUNTVECTORIZER Done in {time() -t0}s")
        print(f"Best score: {gs.best_score_}")
        print("Best parameters set:")             
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(search_space.keys()):
            print(f"\t{param_name}: {best_parameters[param_name]}")
    if not models:
        vec = TfidfVectorizer()
        if topic in ['afrekening_servicekosten', 'gebreken']:
            pipeline = Pipeline([('vec', vec), ('clf', Lasso(max_iter=10000))])
            search_space = {
            'vec__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
            'vec__min_df': (1,2,3),
            'vec__analyzer': ('word', 'char'),
            'vec__norm': ('l1', 'l2', None),
            'clf__normalize': (True, False),
            'clf__selection': ('cyclic', 'random'), 
            }
        elif topic == 'woningverbetering':
            pipeline = Pipeline([('vec', vec), ('clf', LinearRegression())])
            search_space = {
                'vec__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
                'vec__min_df': (1,2,3),
                'vec__analyzer': ('word', 'char'),
                'vec__norm': ('l1', 'l2', None),
                'clf__normalize': (True, False),
            }
        else:
            #Melding, punten, toetsing
            pipeline = Pipeline([('vec', vec), ('clf', ElasticNet(max_iter=10000))])
            search_space = {
            'vec__ngram_range': [(1,1), (1,2), (2,2), (1,3), (2,3), (3,3), (1,4), (2,4), (3,4), (4,4), (1,5), (2,5), (3,5), (4,5), (5,5)],
            'vec__min_df': (1,2,3),
            'vec__analyzer': ('word', 'char'),
            'vec__norm': ('l1', 'l2', None),
            'clf__normalize': (True, False),
            'clf__selection': ('cyclic', 'random'),
        }
        gs = GridSearchCV(pipeline, search_space, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=1)
        labels = data_train['labels']
        X = data_train['text']
        gs.fit(X, labels)
        print(topic.upper())
        print(f"TFIDFVECTORIZER Done in {time() -t0}s")
        print(f"Best score: {gs.best_score_}")
        print("Best parameters set:")             
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(search_space.keys()):
            print(f"\t{param_name}: {best_parameters[param_name]}")
    # final_pipeline = gs.best_estimator_
    # if not models:
    #     final_pipeline.set_params(**best_parameters)   
    # yguess, ytrue = custom_cross_validate(final_pipeline, data_train)
    # print("MAE:", mean_absolute_error(yguess, ytrue))
    # if label != 'per':
    #     print(
    #         f"Predicted percentage error: {predict_percentage(yguess, ytrue, data_train['price'], label)}%")

def draw_plots(yguess, ytrue, topic, phase):
    """Draws a residuals and true vs predicted plot"""
    resids = yguess - ytrue
    fig, ax = plt.subplots(1,2, figsize=(16,9))

    sn.regplot(x=yguess, y=ytrue, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('True vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='True')

    sn.regplot(x=yguess, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    plt.show()
    #plt.savefig('values_and_residuals' + topic +'_' + str(phase) + '.png')

def predict_percentage(y_pred, y_true, price_before, label, topic, phase=False):
    """Predicts the relative error if the relative difference is not predicted directly."""
    y_pred, y_true, price_before = np.array(y_pred), np.array(y_true), np.array(price_before)
    if label == 'abs':
        actual = np.array([abs(y-x)/x for x, y in zip(price_before, y_true)])
        predicted = np.array([abs(y-x)/x for x, y in zip(price_before, y_pred)])
    elif label == 'dif':
        actual = np.array([(y/x) for x, y in zip(price_before, y_true)])
        predicted = np.array([(y/x) for x, y in zip(price_before, y_pred)])
    #draw_plots(predicted, actual)
    #write_errors(topic, phase, predicted, actual)
    return np.mean(abs(actual - predicted)) * 100

def find_good_features(data_train, topic, label):
    """Function that executes the DecisionTreeRegressor"""
    vec = CountVectorizer()
    model = DecisionTreeRegressor(random_state=42, min_samples_leaf=5)
    pipeline = Pipeline([
        ('extract_features', FeatureExtractor()),
        ('union', FeatureUnion(
            transformer_list=[
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text')),
                    ('vec', vec),
                ])),
                ('price', Pipeline([
                    ('selector', ItemSelector(key='price')),
                    ('customtrans', CustomTransformer()),
                    #('scaler', StandardScaler()),
                ])),
                ('city', Pipeline([
                    ('selector', ItemSelector(key='city')),
                    ('customtrans', CustomTransformer()),
                    ('cityt', OneHotEncoder(handle_unknown='ignore')),
                ])),
                ('applicant', Pipeline([
                    ('selector', ItemSelector(key='applicant')),
                    ('customtrans', CustomTransformer()),
                    ('appl', OneHotEncoder()),
                ])),
                ('typewoonruimte', Pipeline([
                    ('selector', ItemSelector(key='typewoonruimte')),
                    ('customtrans', CustomTransformer()),
                    ('typew', OneHotEncoder(handle_unknown='ignore')),
                ])),
                ('lentenant', Pipeline([
                    ('selector', ItemSelector(key='lentenant')),
                    ('customtrans', CustomTransformer()),
                    #('scaler', StandardScaler()),
                ])),
                ('lenlandlord', Pipeline([
                    ('selector', ItemSelector(key='lenlandlord')),
                    ('customtrans', CustomTransformer()),
                    #('scaler', StandardScaler()),
                ])),

            ],
        )),
        ('clf', model)
    ])
    #print(data_train.head)
    labels = data_train['labels']
    X = data_train.drop('labels', axis=1)
    pipeline.fit(X, labels)
    plot_tree(pipeline.named_steps['clf'])
    import graphviz
    feature_names = []
    features_text = pipeline.named_steps['union'].transformer_list[0][1].named_steps['vec'].get_feature_names()
    feature_names.extend(features_text)
    feature_names.extend(['price'])
    feature_names.extend(
        pipeline.named_steps['union'].transformer_list[2][1].named_steps['cityt'].get_feature_names())
    features_applicant = pipeline.named_steps['union'].transformer_list[3][1].named_steps['appl'].get_feature_names()
    feature_names.extend(features_applicant)
    features_typewoonruimte = pipeline.named_steps['union'].transformer_list[4][1].named_steps['typew'].get_feature_names()
    feature_names.extend(features_typewoonruimte)
    feature_names.extend(['lentenant'])
    feature_names.extend(['lenlandlord'])

    dot_data = export_graphviz(pipeline.named_steps['clf'], out_file=None, feature_names=feature_names,)
    graph = graphviz.Source(dot_data)
    graph.render('tree'+topic+label)

def check_assumptions(data_train, topic, label):
    """Function that scatter plots to observe whether the assumptions of the correlation test are met."""
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    data_train.plot.scatter('price', 'labels', ax=axs[0])
    price_levene = levene(data_train['price'], data_train['labels'])
    print('\\multicolumn{3}{c}{', get_topic(topic).upper(), "} \\\\ \\hline")
    fig.suptitle(get_topic(topic).upper())
    print('price &', price_levene.statistic,
          '&', price_levene.pvalue, '\\\\')
    data_train.plot.scatter('lentenant', 'labels', ax=axs[1])
    print('len tenant &', levene(data_train['lentenant'], data_train['labels']).statistic,
          '&', levene(data_train['lentenant'], data_train['labels']).pvalue, '\\\\')
    data_train.plot.scatter('lenlandlord', 'labels', ax=axs[2])
    print('len landlord &', levene(data_train['lenlandlord'], data_train['labels']).statistic,
          '&', levene(data_train['lenlandlord'], data_train['labels']).pvalue, '\\\\ \hline')
    plt.savefig('assumptions'+topic+label+'.png') 


def calculate_correlation(data_train, topic):
    """Calculates the Spearman R correlation between various variables."""
    corr_price = spearmanr(data_train['price'], data_train['labels'])
    corr_lentenant = spearmanr(data_train['lentenant'], data_train['labels'])
    corr_lenlandlord = spearmanr(data_train['lenlandlord'], data_train['labels'])
    print('\\multicolumn{3}{c}{', get_topic(topic).upper(), "} \\\\ \\hline")
    print('price &', corr_price.correlation, '&', corr_price.pvalue, '\\\\')
    print('len tenant &', corr_lentenant.correlation, '&', corr_lentenant.pvalue, '\\\\')
    print('len landlord &', corr_lenlandlord.correlation, '&', corr_lenlandlord.pvalue, '\\\\ \hline')

def simple_regression(data_train, topic):
    """Performs simple regression on various categorical variables."""
    print(topic.upper()) 
    pipeline = Pipeline([('trans', CustomTransformer()), ('onehot', OneHotEncoder(handle_unknown='ignore')), ('reg', LinearRegression())])
    city = cross_validate(pipeline, data_train['city'], data_train['labels'])
    print('City scores:', round(np.mean(city['test_score']),3), round(np.std(city['test_score']),3))
    applicant = cross_validate(pipeline, data_train['applicant'], data_train['labels'])
    print('Applicant scores:', round(np.mean(applicant['test_score']),3), round(np.std(applicant['test_score']),3))
    typewoonruimte = cross_validate(pipeline, data_train['typewoonruimte'], data_train['labels'])
    print('Type woonruimte scores:', round(np.mean(typewoonruimte['test_score']),3), round(np.std(typewoonruimte['test_score']),3))
    pipeline = Pipeline([('trans', CustomTransformer()), ('reg', LinearRegression())])
    price = cross_validate(pipeline, data_train['price'], data_train['labels'])
    print('price', round(np.mean(price['test_score']),3), round(np.std(price['test_score']),3))
    pipeline = Pipeline([('vec', CountVectorizer()), ('reg', LinearRegression())])
    text = cross_validate(pipeline, data_train['text'], data_train['labels'])
    print('text', round(np.mean(text['test_score']),3), round(np.std(text['test_score']),3))


def get_best_label(topic):
    """Helper function to return the best label per category. Values are the result of
    the gridsearch_regression function"""
    label_dict = {
        'afrekening_servicekosten' : 'per',
        'gebreken': 'per',
        'melding': 'per',
        'punten': 'dif',
        'toetsing': 'per',
        'woningverbetering': 'abs',
    }
    return label_dict[topic]


def get_best_params(topic):
    """Returns the best paramters per category. Values are the result of the 
    gridsearch regression function."""
    stop_words = stopwords.words('dutch')
    best_params = {
        'afrekening_servicekosten': {
            'clf__normalize': False,
            'clf__selection': 'random',
            'vec__analyzer': 'char',
            'vec__min_df': 1,
            'vec__ngram_range': (3, 5),
            'vec__norm': None,
            'vec__lowercase': False,
            'vec__stop_words': stop_words,
        },
        'gebreken': {
            'clf__normalize': False,
            'clf__selection': 'random',
            'vec__analyzer': 'char',
            'vec__min_df': 1,
            'vec__ngram_range': (5, 5),
            'vec__norm': None,
            'vec__lowercase': False,
            'vec__stop_words': stop_words,
        }, 
        'melding': {
            'clf__normalize': False,
            'clf__selection': 'random',
            'vec__analyzer': 'word',
            'vec__min_df': 1,
            'vec__ngram_range': (3, 4),
            'vec__norm': None,
            'vec__lowercase': True,
            'vec__stop_words': stop_words,
        }, 
        'punten': {
            'clf__normalize': False,
            'clf__selection': 'random',
            'vec__analyzer': 'char',
            'vec__min_df': 1,
            'vec__ngram_range': (5, 5),
            'vec__norm': None,
            'vec__lowercase': True,
            'vec__stop_words': None,
        }, 
        'toetsing': { 
            'clf__normalize': False,
            'clf__selection': 'random',
            'vec__analyzer': 'word',
            'vec__min_df': 1,
            'vec__ngram_range': (1, 5),
            'vec__norm': None,
            'vec__lowercase': True,
            'vec__stop_words': stop_words,
        }, 
        'woningverbetering': {
            'clf__normalize': True,
            'vec__analyzer': 'word',
            'vec__min_df': 1,
            'vec__ngram_range': (1, 1),
            'vec__lowercase': True,
            'vec__stop_words': stop_words,
        },
    }
    vec = TfidfVectorizer()
    if topic == 'woningverbetering':
        vec = CountVectorizer()
        clf = LinearRegression()
    elif topic in ['afrekening_servicekosten', 'gebreken']:
        clf = Lasso(max_iter=10000)
    else:
        clf = ElasticNet(max_iter=10000)
    pipeline = Pipeline([('vec', vec), ('clf', clf)])
    pipeline.set_params(**best_params[topic])
    return pipeline


def ablation_study(data_train, topic, best_label, model):
    """Performs an ablation study on all the variables used in the thesis.
    Warning: can take a while."""
    transformer_list=[
        ('text', Pipeline([
            ('selector', ItemSelector(key='text')),
            ('vec', get_best_params(topic)[0]),
        ])),
        ('embeddings', Pipeline([
            ('selector', ItemSelector(key='text')),
            ('emb', get_word2vec_embeddings(model, DummyEstimator())[0]),
        ])),
        ('price', Pipeline([
            ('selector', ItemSelector(key='price')),
            ('customtrans', CustomTransformer()),
        ])),
        ('city', Pipeline([
            ('selector', ItemSelector(key='city')),
            ('customtrans', CustomTransformer()),
            ('cityt', OneHotEncoder(handle_unknown='ignore')),
        ])),
        ('applicant', Pipeline([
            ('selector', ItemSelector(key='applicant')),
            ('customtrans', CustomTransformer()),
            ('appl', OneHotEncoder()),
        ])),
        ('typewoonruimte', Pipeline([
            ('selector', ItemSelector(key='typewoonruimte')),
            ('customtrans', CustomTransformer()),
            ('typew', OneHotEncoder(handle_unknown='ignore')),
        ])),
    ]
    all_combinations = []
    for i in range(1,len(transformer_list)+1):
        combinations = list(itertools.combinations(transformer_list, i))
        all_combinations.extend(combinations) 
    print(len(all_combinations))
    labels = data_train['labels']
    X = data_train.drop('labels', axis=1)
    for trans_list in all_combinations[26:]:
        pipeline = Pipeline([
            ('extract_features', FeatureExtractor()),
            ('union', FeatureUnion(transformer_list=trans_list, n_jobs=-1)),
            ('clf', get_best_params(topic)[1])
        ])        
        print(topic.upper(), trans_list)
        scores = cross_val_score(pipeline, X, labels, scoring='neg_mean_absolute_error', n_jobs=-1)
        #labels, yguess = custom_cross_validate(pipeline, data_train)
        #print("MAE:", mean_absolute_error(yguess, labels))
        # print('R2 score:', r2_score(labels, yguess))
        print('MAE scores from each iteration:', scores)
        print('Average K-Fold MAE Score: ', np.mean(scores))
        print('Std deviation:', np.std(scores))
        if best_label != 'per':
            yguess = cross_val_predict(pipeline, X, labels, n_jobs=-1)   
            print(
                f"Predicted percentage error: {predict_percentage(yguess, labels, data_train['price'], best_label, topic)}%")


def final_model(data_train, data_test, topic, phase, best_label, model):
    """Returns the scores and plots for the final models"""
    transformer_list=[
        ('text', Pipeline([
            ('selector', ItemSelector(key='text')),
            ('vec', get_best_params(topic)[0]),
        ])),
        ('embeddings', Pipeline([
            ('selector', ItemSelector(key='text')),
            ('emb', get_word2vec_embeddings(model, DummyEstimator())[0]),
        ])),
        ('price', Pipeline([
            ('selector', ItemSelector(key='price')),
            ('customtrans', CustomTransformer()),
        ])),
        ('city', Pipeline([
            ('selector', ItemSelector(key='city')),
            ('customtrans', CustomTransformer()),
            ('cityt', OneHotEncoder(handle_unknown='ignore')),
        ])),
        ('applicant', Pipeline([
            ('selector', ItemSelector(key='applicant')),
            ('customtrans', CustomTransformer()),
            ('appl', OneHotEncoder()),
        ])),
        ('typewoonruimte', Pipeline([
            ('selector', ItemSelector(key='typewoonruimte')),
            ('customtrans', CustomTransformer()),
            ('typew', OneHotEncoder(handle_unknown='ignore')),
        ])),
    ]
    if topic == 'afrekening_servicekosten':
        indices = [0,2,3,5]
    elif topic == 'gebreken':
        indices = [0,1]
    elif topic == 'melding':
        indices = [2,3]
    elif topic == 'punten':
        indices = [0,2]
    elif topic == 'toetsing':
        indices = [0,2,5]
    elif topic == 'woningverbetering':
        indices = [2,4]
    features = [transformer_list[i] for i in indices]
    ytrain = data_train['labels']
    Xtrain = data_train.drop('labels', axis=1)
    ytest = data_test['labels']
    Xtest = data_test.drop('labels', axis=1)
    pipeline = Pipeline([
        ('extract_features', FeatureExtractor()),
        ('union', FeatureUnion(transformer_list=features, n_jobs=-1)),
        ('clf', get_best_params(topic)[1])
    ])        
    print('before fit')
    pipeline.fit(Xtrain, ytrain)
    print('after fit')
    feature_names = []
    if topic in ['afrekening_servicekosten', 'gebreken', 'punten', 'toetsing']:
        features_text = pipeline.named_steps['union'].transformer_list[indices.index(0)][1].named_steps['vec'].get_feature_names()
        feature_names.extend(features_text)
    if topic == 'gebreken':
        feature_names.extend(200*['embedding'])
    else:
        feature_names.extend(['price'])
    if topic in ['afrekening_servicekosten', 'melding']:
        feature_names.extend(
            pipeline.named_steps['union'].transformer_list[indices.index(3)][1].named_steps['cityt'].get_feature_names())
    if topic == 'woningverbetering':
        features_applicant = pipeline.named_steps['union'].transformer_list[indices.index(4)][1].named_steps['appl'].get_feature_names()
        feature_names.extend(features_applicant)
    if topic in ['afrekening_servicekosten', 'toetsing']:
        features_typewoonruimte = pipeline.named_steps['union'].transformer_list[indices.index(5)][1].named_steps['typew'].get_feature_names()
        feature_names.extend(features_typewoonruimte)
    if topic not in ['melding', 'woningverbetering']:
        plot_coefficients(pipeline.named_steps['clf'], feature_names, topic)
    yguess = pipeline.predict(Xtest)
    print("MAE:", mean_absolute_error(yguess, ytest))
    draw_plots(yguess, ytest, topic, phase)
    if best_label != 'per':
        print(
            f"Predicted percentage error: {predict_percentage(yguess, ytest, data_test['price'], best_label, topic, phase=phase)}%")
    else:
        pass
        #write_errors(topic, phase, yguess, ytest)

def write_errors(topic, phase, yguess, ytest):
    """Writes the MAE scores to file so they are easily inputted into R"""
    with open(f'../significance_testing/{topic}_{str(phase)}.txt', 'w') as out:
            out.write('Error\n')
            for i, j in zip(yguess, ytest):
                out.write(f'{j-i}\n')

def baseline(data_train, data_test, best_label, topic, phase):
    """Calculates the MAE for the baseline by guessing the mean for every
    category."""
    ytrain = data_train['labels']
    Xtrain = data_train.drop('labels', axis=1)
    ytest = data_test['labels']
    Xtest = data_test.drop('labels', axis=1)
    dummy_regressor = DummyRegressor(strategy="mean")
    dummy_regressor.fit(Xtrain, ytrain)
    yguess = dummy_regressor.predict(Xtest)
    
    print("Baseline MAE:", mean_absolute_error(yguess, ytest))
    if best_label != 'per':
        print(
            f"Baseline predicted percentage error: {predict_percentage(yguess, ytest, data_test['price'], best_label, topic, phase=phase)}%")

def main():
    """Main function to bring everything together. Function calls are sometimes commented if they were
    not used the last time we ran the experiment."""
    args = parse_args()
    if args.topic == 'all':
        topics = ['afrekening_servicekosten', 'gebreken', 'melding', 'punten', 'toetsing', 'woningverbetering']
    if args.topic == 'all' and args.dates == True:
        topics = ['afrekening_servicekosten', 'gebreken', 'punten', 'toetsing']
    if 'topics' not in locals():
        topics = [args.topic]
    
    #cbow200 = KeyedVectors.load_word2vec_format("../embeddings/custom_w2v_200d_cbow.txt", binary=False)
    cbow300 = KeyedVectors.load_word2vec_format("../embeddings/custom_w2v_300d_cbow.txt", binary=False)
    #skip200 = KeyedVectors.load_word2vec_format("../embeddings/custom_w2v_200d_skipgram.txt", binary=False)
    for topic in topics:
        if args.phase == 'all':
            train_json = open_json_files(topic, False, 'train')
            old_test = open_json_files(topic, False, 'test')
            train_json = train_json + old_test
            test_json = open_json_files(topic, True, 'test')
        else:
            train_json = open_json_files(topic, args.dates, 'train')
            test_json = open_json_files(topic, args.dates, 'test')
        best_label = get_best_label(topic)
        data_train, data_test = extract_features(train_json, test_json, topic, best_label)
        #find_good_features(data_train, topic, best_label)
        #check_assumptions(data_train, topic, best_label)
        #calculate_correlation(data_train, topic)
        #gridsearch_regression(data_train, topic, best_label, models=False)
        print(topic)
        if args.phase == 'dev':
            if topic in ['melding', 'punten', 'woningverbetering']:
                model = cbow200
            elif topic in ['gebreken, toetsing']:
                model = cbow300
            else:
                model = skip200
            # simple_regression(data_train, topic)
            ablation_study(data_train, topic, best_label, model)
            
        else:
            phase = args.phase + str(args.dates)
            model = cbow300
            print(topic.upper())
            if args.baseline:
                baseline(data_train, data_test, best_label, topic, phase)
            final_model(data_train, data_test, topic, phase, best_label, model)

if __name__ == "__main__":
    main()
