#!/usr/bin/python3
import sys
sys.path.append('../')
from utils import open_csv_files, open_csv_files_and_check_date
from lxml import html
import requests 
import os
import pandas as pd


def download_files(case_numbers, future=False):
    """Downloads cases from the Rent Tribunal website"""
    folder = 'all_cases'
    if future:
        folder = 'future_cases'
    for case_number in case_numbers:
        if not os.path.isfile(f'../data/{folder}/{case_number}.pdf'):
            pdf = requests.get(f'https://huurcommissie.nl/data/or_files/{case_number}.pdf')
            if pdf.status_code == 200:
                with open(f'../data/{folder}/{case_number}.pdf', 'wb') as infile:
                    infile.write(pdf.content)
            else:
                print(f'No file found for case {case_number}')


case_numbers = open_csv_files(['2015', '2016','2017','2018','2019','2020'])
download_files(case_numbers)

future_cases = open_csv_files_and_check_date()
#convert to datetime so we can sort
future_cases['Datum afdoening'] = pd.to_datetime(future_cases['Datum afdoening'], format='%d-%m-%y')
future_cases = future_cases.sort_values(by='Datum afdoening')
filter_date = '21-01-2020'
a = future_cases[future_cases['Datum afdoening'] > pd.to_datetime(filter_date)]
a = a.sort_values(by=['Datum afdoening'])
case_numbers = a['Zaaknummer'].tolist()
download_files(case_numbers, future=True)


