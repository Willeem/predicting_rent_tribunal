#!/usr/bin/python3 
import re
import sys
import json
import nltk
import multiprocessing
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.scripts import glove2word2vec
# from glove import Corpus, Glove
sys.path.append('../')
from utils import list_files_in_dir, open_and_process_pdf_file


def clean_text(text, case_number):
    """Clean page and case numbers"""
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
    return without_case_nr


def doc2vec_trainer(texts):
    """Trains doc2vec embeddings, refer to the gensim documentation for explanations."""
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(), epochs=20)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save('doc2vec_300_dim_mincount2.txt')
    model = Doc2Vec(documents, vector_size=200, window=5, min_count=5, workers=multiprocessing.cpu_count(), epochs=20)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save('doc2vec_200dim_mincount2.txt')


def fasttext_trainer(sentences):
    """Trains fasttext embeddings, refer to the gensim documentation for explanations."""
    model = FastText(size=200, window=5, min_count=5)
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=20)
    model.wv.save('fasttext_200d25e.txt')
    model = FastText(size=300, window=5, min_count=5)
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=20)
    model.wv.save('fasttext_300d25e.txt')


def word2vec_trainer(sentences):
    """Trains word2vec embeddings, refer to the gensim documentation for explanations."""
    w2v = Word2Vec(sentences, size=300, window=10, min_count=5, negative=15, 
        iter=10, workers=multiprocessing.cpu_count(), sg=1)
    w2v.wv.save_word2vec_format('custom_w2v_300d_skipgram.txt')
    w2v = Word2Vec(sentences, size=200, window=10, min_count=5, negative=15,
                   iter=10, workers=multiprocessing.cpu_count(), sg=1)
    w2v.wv.save_word2vec_format('custom_w2v_200d_skipgram.txt')
    w2v = Word2Vec(sentences, size=300, window=5, min_count=5, negative=15,
                   iter=10, workers=multiprocessing.cpu_count(), sg=0)
    w2v.wv.save_word2vec_format('custom_w2v_300d_cbow.txt')
    w2v = Word2Vec(sentences, size=200, window=5, min_count=5, negative=15,
                   iter=10, workers=multiprocessing.cpu_count(), sg=0)
    w2v.wv.save_word2vec_format('custom_w2v_200d_cbow.txt')


def get_cases():
    """Strips all cases from case numbers and puts them in json file"""
    files = list_files_in_dir('../data/all_cases/')
    texts = [clean_text(open_and_process_pdf_file(filename), filename)
             for filename in files]
    with open('all_texts_stripped_from_casenumbers.json', 'w') as outfile:
        json.dump(texts, outfile)
    
def get_policy_books():
    """Combines all policy books in a json file"""
    files = list_files_in_dir('../data/policy_books/')
    texts = [open_and_process_pdf_file(filename)
             for filename in files]
    with open('all_policy_books.json', 'w') as outfile:
        json.dump(texts, outfile)
def main():
    get_cases()
    get_policy_books()
    with open('all_texts_stripped_from_casenumbers.json','r') as infile:
        cases = json.load(infile)
    
    with open('all_policy_books.json', 'r') as infile:
        policy_books = json.load(infile)
    sentences, docs = [], []
    texts = policy_books + cases
    for text in texts:
        doc = []
        for sent in nltk.sent_tokenize("".join(text)):
            sentences.append(nltk.word_tokenize(sent)) 
            doc.extend(nltk.word_tokenize(sent))
        docs.append(doc)
    word2vec_trainer(sentences)
    fasttext_trainer(sentences)
    doc2vec_trainer(docs)
if __name__ == "__main__":
    main()
