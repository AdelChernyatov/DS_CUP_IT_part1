#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import re
import nltk
from nltk import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')



def clean_second(t):
	t = re.sub(r'&#\d+(;)?','',t)
	t = re.sub(r'#\d+(;)?','',t)
	t = re.sub('\^\w','',t)
	t = re.sub('(\t|\s{2,})', ' ', t) #двойные пробелы и табуляции
	t = re.sub('\s{2,}', ' ', t) #двойные пробелы еще раз
	#t = re.sub('#\w+', '', t)
	
	#обнаружение слипшихся частей текста (сразу после точки буква, а не пробел)
	t_new = ''
	
	inds = [x.start() for x in re.finditer(r'(\w|\s|\.)[?!,\.]\w',t)]
	for i in range(len(inds)):
		if i==0:
			t_new += (t[:inds[i]+2] + ' ')
		else:
			t_new += (t[inds[i-1]+2:inds[i]+2] + ' ')
			
	try:
		t_new += t[inds[i]+2:]
		t = t_new
	except:
		pass
		
	t_new = ''
    	#обнаружение слипшихся частей текста (сразу после : буква, а не пробел)
	t_new = ''
	
	inds = [x.start() for x in re.finditer(r'\w\:\w',t)]
	for i in range(len(inds)):
		if i==0:
			t_new += (t[:inds[i]+2] + ' ')
		else:
			t_new += (t[inds[i-1]+2:inds[i]+2] + ' ')
			
	try:
		t_new += t[inds[i]+2:]
		t = t_new
	except:
		pass
		
	t_new = ''
	
	#обнаружение слипшихся левых скобочек в тексте
	inds = [x.start() for x in re.finditer(r'\w\(',t)]
	for i in range(len(inds)):
		if i==0:
			t_new += (t[:inds[i]+1] + ' ')
		else:
			t_new += (t[inds[i-1]+1:inds[i]+1] + ' ')

	try:
		t_new += t[inds[i]+1:]
		t = t_new
	except:
		pass
	
	#обнаружение слипшихся правых скобочек в тексте
	t_new = ''

	inds = [x.end() for x in re.finditer(r'\)\w',t)]
	for i in range(len(inds)):
		if i==0:
			t_new += (t[:inds[i]-1] + ' ')
		else:
			t_new += (t[inds[i-1]-1:inds[i]-1] + ' ')

	try:
		t_new += t[inds[i]-1:]
		t = t_new
	except:
		pass
    
    #работаем с разъехавшимися сайтами
	t = re.sub('\. com\s', '.com ', t)
	t = re.sub('\. net\s', '.net ', t)
	t = re.sub('\. en\s', '.en ', t)
	t = re.sub('\. uk\s', '.uk ', t)
	t = re.sub('\. ru\s', '.ru ', t)
	t = re.sub('\. us\s', '.us ', t)
	t = re.sub('\. au\s', '.au ', t)
	t = re.sub('www\. \s', 'www.', t)
    
	return t

def clean_first(text):
    '''
    '''
    text = text.lower()
    text = text.replace("..", ".")
    text = text.replace("•", "")
    text = text.replace("--", "")
    text = text.replace("@", "")
    text = text.replace("~", "")    
    text = text.replace("´", "")
    text = text.replace("‘", "”")
    text = text.replace("’", "”")
    text = text.replace("\"", "")
    text = text.replace("[", "")
    text = text.replace("]","")
    text = text.replace("{","")
    text = text.replace("}", "")
    text = text.replace("|", "")
    text = text.replace("$", "")
    text = text.replace("+", "")
    text = text.replace("*", "")
    text = text.replace("%", "")
    text = text.replace("#", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace("_", " _ ")
    text = text.replace("_", "")
    text = text.replace("  ", " ")
    text = text.replace("   ", " ")
    text = text.replace("\'", "")       

    text = text.replace("ive", "i have")
    text = text.replace("id", "i would")
    text = text.replace("youd", "you would")
    text = text.replace("wed", "we had")
    text = text.replace("theyd", "they had")
    text = text.replace("theyre", "they are")
    text = text.replace("im", "i not")
    text = text.replace("youve", "you have")
    text = text.replace("weve", "we have")
    text = text.replace("ain't", "am not")
    text = text.replace("aren't", "are not")
    text = text.replace("can't", "cannot")
    text = text.replace("can't've", "cannot have")
    text = text.replace("'cause", "because")
    text = text.replace("could've", "could have")
    text = text.replace("couldn't", "could not")
    text = text.replace("couldn't've", "could not have")
    text = text.replace("should've", "should have")
    text = text.replace("should't", "should not")
    text = text.replace("should't've", "should not have")
    text = text.replace("would've", "would have")
    text = text.replace("would't", "would not")
    text = text.replace("would't've", "would not have")
    text = text.replace("didn't", "did not")
    text = text.replace("isn't", "is not")
    text = text.replace("doesn't", "does not")
    text = text.replace("don't", "do not")
    text = text.replace("hadn't", "had not")
    text = text.replace("hadn't've", "had not have")
    text = text.replace("hasn't", "has not")
    text = text.replace("haven't", "have not")
    text = text.replace("haven't", "have not")
    text = text.replace("haven't", "have not")
    text = text.replace("haven't", "have not")
    text = text.replace("he'd", "he would")
    text = text.replace("haven't", "have not")
    text = text.replace("he'd've", "he would have")
    text = text.replace("she'd've", "she would have")
    text = text.replace("aint", "am not")
    text = text.replace("arent", "are not")
    text = text.replace("cant", "cannot")
    text = text.replace("cant've", "cannot have")
    text = text.replace("'cause", "because")
    text = text.replace("couldve", "could have")
    text = text.replace("couldnt", "could not")
    text = text.replace("couldntve", "could not have")
    text = text.replace("shouldve", "should have")
    text = text.replace("shouldt", "should not")
    text = text.replace("shouldtve", "should not have")
    text = text.replace("wouldve", "would have")
    text = text.replace("wouldt", "would not")
    text = text.replace("wouldtve", "would not have")
    text = text.replace("didnt", "did not")
    text = text.replace("isnt", "is not")    
    text = text.replace("doesnt", "does not")
    text = text.replace("dont", "do not")
    text = text.replace("hadnt", "had not")
    text = text.replace("hadntve", "had not have")
    text = text.replace("hasnt", "has not")
    text = text.replace("havent", "have not")
    text = text.replace("havent", "have not")
    text = text.replace("havent", "have not")
    text = text.replace("havent", "have not")
    text = text.replace("hed", "he would")
    text = text.replace("shed", "she would")
    text = text.replace("havent", "have not")
    text = text.replace("hedve", "he would have")
    text = text.replace("shedve", "she would have")
    text = text.replace("'s", "")
    text = text.replace("'t", "")
    text = text.replace("'ve", "")

    return text

def clean_third(text):
    '''
    '''
    text = text.lower()
    text = text.replace("  ..", ".")
    text = text.replace("..", ".")
    text = text.replace("...", ".")
    text = text.replace("•", "")
    text = text.replace("--", "")
    text = text.replace("@", "")
    text = text.replace("~", "")
    text = text.replace("\'", "") 
    text = text.replace(".", " . ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace(";", " ; ")
    text = text.replace(":", " : ")
    text = text.replace(",", " , ")    
    text = text.replace("´", "")
    text = text.replace("‘", "”")
    text = text.replace("’", "”")
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("–", "")
    text = text.replace("—", "")
    text = text.replace("[", "")
    text = text.replace("]","")
    text = text.replace("{","")
    text = text.replace("}", "")
    text = text.replace("|", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("$", "")
    text = text.replace("+", "")
    text = text.replace("*", "")
    text = text.replace("%", "")
    text = text.replace("#", "")
    text = text.replace("\n", " \n ")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace("_", " _ ")
    text = text.replace("_", "")
    text = text.replace("  ", " ")
    text = text.replace("   ", " ")
    text = text.replace("boost", " ")
    
    text = ''.join([i for i in text if not i.isdigit()])
    
    return text

def count_of_smiles(text):
    cur_count = 0
    for i in list_of_smiles:
        cur_count += text.count(i)
    return cur_count

def split_word( word):
    result = list()
    while(len(word) > 2):
        i = 1
        found = True
        while(found):
            i = i + 1
            synsets = wn.synsets(word[:i])
            for s in synsets:
                if edit_distance(s.name().split('.')[0], word[:i]) == 0:
                    found = False
                    break;
        result.append(word[:i])
        word = word[i:]
    print(result)
    
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.
    
def lem_pos(text):
    '''
    '''
    wnl = WordNetLemmatizer()
    
    text_lem = [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(word_tokenize(text))]

    return text_lem

def delete_stopwords(review):
    
    review =[word for word in review if word not in stoplist_combined] #removing stop words
  
    text = ' '.join(review)
  
    return text

