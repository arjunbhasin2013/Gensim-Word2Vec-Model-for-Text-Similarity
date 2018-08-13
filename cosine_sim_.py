from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec	

sample = open('/home/user/Desktop/alice.txt')
s = sample.read()


#~ print(sample.read())

f = s.replace("\n", " ")

data = []

for i in sent_tokenize(f):
	temp = []
	
	for j in word_tokenize(i):
		temp.append(j.lower())
		
	data.append(temp)

#~ print(len(data))

#### Word2Vec Model

model_1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5)

#### Printing Results

print("Similarity : ", model_1.similarity('books','rabbit'))
