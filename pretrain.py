import json
from gensim.models import FastText

def loadData(fileName):
	with open(fileName, 'r') as f:
		data = f.readlines()
	return data
