import json
from gensim.models import FastText

def loadData(fileName, fields):
	
	with open(fileName, 'r') as f:
		dataset = f.readlines()
		
	corpus = []
	for i in range(len(dataset)):
		datum = json.loads(dataset[i])
		for field in fields:
			corpus.append(datum[field])
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

def wv(*datasets, clean = False, trim = False, spacy = False):
	data = []
	for (fileName, fields) in datasets:
		data += loadData(fileName, fields)
	if clean:
		data = [clean(datum) for datum in data]
	return data
