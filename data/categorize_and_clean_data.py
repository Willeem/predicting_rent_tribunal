from collections import defaultdict
from shutil import copyfile
import os
import pandas as pd
import sys
import json
import random
import re
from datetime import datetime
sys.path.append('../')
from utils import list_files_in_dir, open_and_process_pdf_file, open_csv_files_in_df, write_to_json, make_folder, open_json_files


def open_csv_with_categories(filename=None):
    """Returns a csv with categories as keys and a list of filenames as values."""
    category_dict = defaultdict(list)
    df = open_csv_files_in_df(filename=filename)
    for index, row in df.iterrows():
        if 'afrekening' in row['Wetsartikel'].lower() and 'servicekosten' in row['Wetsartikel'].lower():
            category_dict['afrekening_servicekosten'].append(index)
        elif 'woningverbetering' in row['Wetsartikel'].lower():
            category_dict['woningverbetering'].append(index)
        elif 'huurverlaging' in row['Wetsartikel'].lower() and 'gebreken' in row['Wetsartikel'].lower():
            category_dict['gebreken'].append(index)
        elif 'toetsing' in row['Wetsartikel'].lower():
            category_dict['toetsing'].append(index)
        elif 'punten' in row['Wetsartikel'].lower() or 'voorstel huurprijsverlaging' in row['Wetsartikel'].lower():
            category_dict['punten'].append(index)
        elif 'melding' in row['Wetsartikel'].lower():
            category_dict['melding'].append(index)
        else:
            pass
    return category_dict


def make_folders_and_place_files(case_dict):
    """Creates folders from keys in the dictionary and places files in one large folder."""
    files_not_foundlist = []
    for item in case_dict:
        for case in case_dict[item]:
            directory = os.getcwd() + '/' + item + '/'
            path = directory + str(case) + '.pdf'
            original_path = os.getcwd() + '/all_cases/' + str(case) + '.pdf'
            make_folder(directory)
            try:
                copyfile(original_path, path)
            except FileNotFoundError:
                files_not_foundlist.append(case)
    return files_not_foundlist


def check_informativity(text, case_number):
    """Used to remove cases that only contain one sentence explaining that the case is treated in another document"""
    for i in text:
        if 'is verwerkt in zaak:' in i:
            return False
    return True


def combine_cases_and_check_informativity():
    """All cases are checked for whether they actually contain information or just point to another case.
    Returns a dictionary with category as key and useful documents as values."""
    subject_list = ['afrekening_servicekosten', 'gebreken', 
    'melding', 'punten', 'toetsing', 'woningverbetering']
    all_files = [list_files_in_dir(os.getcwd() + '/' + item) for item in subject_list]
    flat_files = [item for sublist in all_files for item in sublist]
    all_years = open_csv_files_in_df()
    every_case_by_subject = defaultdict(list)
    for f in flat_files:
        case_number = f.split('/')[-1][:-4]
        category = f.split('/')[-2]
        text = open_and_process_pdf_file(f)
        if check_informativity(text,case_number):
            #Useful document 
            try:
                case_details = {k:v for k,v in all_years.loc[case_number].to_dict().items() if pd.notnull(v)}
            except KeyError:
                case_details = {k:v for k,v in all_years.loc[int(case_number)].to_dict().items() if pd.notnull(v)}
            case_details['text'] = clean_text(text, str(case_number))
            case_details['Zaaknummer'] = str(case_number)
            every_case_by_subject[category].append(case_details)
        else:
            pass
    return every_case_by_subject, all_years


def split_train_test(df, cases, future=False):
    """Randomly samples cases if future=False, else splits cases on date and writes them to train.json and test.json"""
    folder = '/future/' if future else '/shuffle/'
    make_folder(os.getcwd() + folder)
    for category in cases.keys():
        if not future:
            random.seed(42)
            random.shuffle(cases[category])
            case_list = cases[category]
        else:
            case_list = sorted(cases[category], key = lambda i: datetime.strptime(i['Datum afdoening'], '%d-%m-%y'))
        case_dict = {}
        case_dict['train'] = case_list[:int(.80*len(case_list))]
        #case_dict['dev'] = case_list[int(.70*len(case_list)):int(.80*len(case_list))]
        case_dict['test'] = case_list[int(.80*len(case_list)):]
        for _type in case_dict.keys():
            path = os.getcwd() + folder + category + '/' 
            make_folder(path)
            write_to_json(path + _type, case_dict[_type])

    
def clean_text(text, case_number):
    """Clean page numbers as well as info that is not available before the case"""
    zkn = False
    for item in text[:]:
        if item.strip().lower().startswith('pagina') or item.strip().lower().startswith('oru'):
            text.remove(item)
        elif '-----' in item:
            text.remove(item)
        elif 'ZKN' in case_number:
            zkn = True
    
    if zkn:
        #Remove document number on each page:
        without_page_nr = re.sub(r'C\s+?O\s+?U.*? -[0-9]+-[0-9]+', '', " ".join(text))
        #Remove case number on each page:
        without_case_nr = re.sub(r'Zaaknummer .*?Datum .*? [0-9][0-9][0-9][0-9] .*?-[0-9]+-[0-9]+', '', without_page_nr)
        
    else:
        #Newer cases need another regex for case number removal:
        without_case_nr = re.sub(r'Zaaknummer\s+?[0-9]+', '', " ".join(text))
    if 'Beoordeling' not in without_case_nr:
        print(f"No beoordeling: {text}")
    #Remove 'Kern van de Uitspraak':
    without_kern = re.sub(r'Kern van de uitspraak .*(?=I Verloop)', '', without_case_nr)
    return without_kern

def process_future_cases(case_dict):
    """Processes future cases for the Test2 test set."""
    df = open_csv_files_in_df('../csvs/future_cases_2020.csv', dtype=str)
    every_case_by_subject = defaultdict(list)
    for key in case_dict.keys():
        if len(case_dict[key]) > 50:
            for item in case_dict[key]:
                try:
                    text = open_and_process_pdf_file('future_cases/'+str(item)+'.pdf')
                except FileNotFoundError:
                    print(f"{item} was not found.")
                    continue
                if check_informativity(text,item):
                    #Useful document 
                    try:
                        case_details = {k:v for k,v in df.loc[item].to_dict().items() if pd.notnull(v)}
                    except KeyError:
                        case_details = {k:v for k,v in df.loc[int(item)].to_dict().items() if pd.notnull(v)}
                    case_details['text'] = clean_text(text, str(item))
                    case_details['Zaaknummer'] = str(item)
                    if datetime.strptime(case_details['Datum afdoening'],'%d-%m-%y') > datetime.strptime('21-01-20', '%d-%m-%y'):
                        every_case_by_subject[key].append(case_details)
                else:
                    pass
    
    for key in every_case_by_subject:    
        path = os.getcwd() + '/future_new/' + key + '/' 
        make_folder(path)
        write_to_json(path + 'test_future', every_case_by_subject[key])
def main():
    """Used to call the right functions at appropriate times."""
    case_dict = open_csv_with_categories()
    print(make_folders_and_place_files(case_dict))
    every_case_by_subject, all_years = combine_cases_and_check_informativity()
    split_train_test(all_years, every_case_by_subject)
    case_dict = open_csv_with_categories(filename='../csvs/future_cases_2020.csv')
    process_future_cases(case_dict)
       




if __name__ == "__main__":
    main()
