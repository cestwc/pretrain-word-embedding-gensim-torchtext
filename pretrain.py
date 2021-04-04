import re
import json
import spacy
from gensim.models import FastText

def loadData(fileName, fields):
	
	with open(fileName, 'r') as f:
		dataset = f.readlines()
		
	corpus = []
	for i in range(len(dataset)):
		datum = json.loads(dataset[i])
		for field in fields:
			corpus.append(datum[field].strip())
			
	return corpus

def clean(text):
	
	text = str(text)
	text = text.lower()
	text = re.sub(r'\'s',r'\tis',text)
	text = re.sub(r'\'ll',r'\twill',text)
	text = re.sub(r'\'m',r'\tam',text)
	text = re.sub(r'\'re',r'\tare',text)
	text = re.sub(r'\'d',r'\twould',text)
	text = re.sub(r'n\'t',r'\tnot',text)
	text = re.sub('[^a-zA-Z0-9]',' ',text) 
	text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
	
	return text

def trim(tokens, trim_length):
	return tokens if len(tokens) <= trim_length else tokens[0:trim_length]

def wv(*datasets, directory = 'emb.txt', clean_text = False, tokenizer = None, trim_length = None, algo = 'fasttext', size = 128, window = 3, min_count = 2, workers = 4, sg = 1, iter = 1500):
	
	data = []
	for (fileName, fields) in datasets:
		data += loadData(fileName, fields)
		
	if clean_text:
		data = [clean(datum) for datum in data]
		
	if tokenizer == None:
		corpus = [datum.split() for datum in data]
	else:
		spacy_en = spacy.load('en_core_web_sm')
		corpus = [[tok.text for tok in spacy_en.tokenizer(datum)] for datum in data]
		
	if isinstance(trim_length, int):
		corpus = [trim(x, trim_length) for x in corpus]
		
	if algo == 'fasttext':
		model = FastText(corpus, size = size, window=window, min_count=min_count, workers=workers, sg=sg, iter = iter)
	else:
		model = Word2Vec(corpus, size = size, window=window, min_count=min_count, workers=workers, sg=sg, iter = iter)
	
	model.wv.save_word2vec_format(directory)
	
	return model
