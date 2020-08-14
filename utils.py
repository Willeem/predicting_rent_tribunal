#!/usr/bin/python3
import os
from os import listdir 
from os.path import isfile, join
from tika import unpack
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import json


def calculate_label(x, y, topic):
    """Helper function to create labels for classification"""
    if topic in ['afrekening_servicekosten', 'gebreken', 'punten', 'toetsing']:        
        if y < x:
            return 'lower'
        return 'same or higher'
    else:
        if y > x:
            return 'higher'
        return 'same or lower'

def list_files_in_dir(direc):
    """Returns all files in a given directory"""
    return [direc + '/' + f for f in listdir(direc) if isfile(join(direc, f))]

def open_csv_files(years):
    """Returns case numbers for the given years."""
    case_numbers = []
    for year in years:
        with open('../csvs/'+year+'-exportList.csv', 'r') as infile:
            for line in infile:
                line = line.split(';')
                case_number = line[0]
                case_numbers.append(case_number)
    return case_numbers

def open_csv_files_and_check_date():
    """Returns case numbers for the future cases csv."""
    case_numbers = pd.read_csv('../csvs/future_cases_2020.csv', delimiter=';')
    return case_numbers

def open_csv_files_in_df(filename=None, dtype=str):
    """Returns a dataframe with every column from the csv as key"""
    if filename:
        return pd.read_csv(filename, delimiter=';', index_col='Zaaknummer', dtype=str)
    frames = []
    for year in ['2015','2016','2017','2018','2019','2020']:
        df = pd.read_csv('../csvs/'+year+'-exportList.csv', delimiter=';', index_col='Zaaknummer')
        frames.append(df)
    return pd.concat(frames)

def open_and_process_pdf_file(f):
    """Returns list of text."""
    rawText = unpack.from_file(f)
    rawList = rawText['content'].splitlines()
    return rawList

def open_json_files(topic, dates, phase):
    """Opens json files and returns a dictionary."""
    folder = '/future_new/' if dates and phase == 'test' else '/shuffle/'
    with open('../data' + folder + topic + f'/{phase}.json', 'r') as infile:
        data = json.load(infile)
    return data

def make_folder(directory):
    """Helper function to create a directory"""
    if not os.path.exists(directory):
        os.mkdir(directory)

def reformat_price(x):
    """Reformats the price from 'natural language' float way of representation (e.g. 10.000,50 to 10000.50)"""
    return float(x.replace('.', '').replace(',', '.'))

def write_to_json(path, dictionary):
    """Helper function to write dictionaries to json files"""
    with open(path+'.json', 'w') as outfile:
        json.dump(dictionary,outfile)

def get_topic(topic):
    """Returns english translation of the categories of the rent tribunal"""
    topics = {
        'afrekening_servicekosten': 'Settlement of service costs',
        'gebreken': 'Decrease in rent because of maintenance problems',
        'melding': 'Mention repairs of maintenance problems',
        'punten': 'Decrease in rent because of points',
        'toetsing': 'Assessment of initial rent',
        'woningverbetering': 'Increase in rent after home improvements',

    }
    return topics[topic]

def plot_coefficients(classifier, feature_names, topic, top_features=20):
    """Plots coefficients of linear machine learning models."""
    coef = np.ravel(classifier.coef_)
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # create plot
    plt.figure(figsize=(15, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig('coefficients_'+topic+'.png')

class FeatureExtractor(TransformerMixin, BaseEstimator):
    """Helper function to extract features from a dataframe"""
    def fit(self, x, y=None):
        return self 
    
    def transform(self, data):
        return data
if __name__ == "__main__":
    print('hello')