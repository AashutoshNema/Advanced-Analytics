# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:52:10 2019

@author: aashu
"""

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

#Import Text File
#adoc = []
#for i in range(1,8):
    #with open ("TextFiles/T%d.txt"%(i), "r") as text_file:
     #   adoc.append(text_file.read())

with open ("TextFiles/T8.txt", "r") as text_file:
    adoc = text_file.read()

# Convert to all lower case - required
a_discussion = ("%s" %adoc).lower()

        
# Remove unwanted punctuation
a_discussion = a_discussion.replace('-', ' ')
a_discussion = a_discussion.replace('_', ' ')
a_discussion = a_discussion.replace(',', ' ')
a_discussion = a_discussion.replace("'nt", " not")

# Tokenize
tokens = word_tokenize(a_discussion)
tokens = [word.replace(',', '') for word in tokens]
tokens = [word for word in tokens if ('*' not in word) and \
("''" != word) and ("``" != word) and \
(word!='description') and (word !='dtype') \
and (word != 'object') and (word!="'s")]
print("\nDocument contains a total of", len(tokens), " terms.")
token_num = FreqDist(tokens)
for pos, frequency in token_num.most_common(20):
    print('{:<15s}:{:>4d}'.format(pos, frequency))


    
#POS Tagging
tagged_tokens = nltk.pos_tag(tokens)
pos_list = [word[1] for word in tagged_tokens if word[1] != ":" and \
word[1] != "."]
pos_dist = FreqDist(pos_list)
pos_dist.plot(title="Parts of Speech")
for pos, frequency in pos_dist.most_common(pos_dist.N()):
    print('{:<15s}:{:>4d}'.format(pos, frequency))

# Removing stop words
stop = stopwords.words('english') + list(string.punctuation)
stop_tokens = [word for word in tagged_tokens if word[0] not in stop]
# Removing single character words and simple punctuation
stop_tokens = [word for word in stop_tokens if len(word) > 1]
# Removing numbers and possive "'s"
stop_tokens = [word for word in stop_tokens \
if (not word[0].replace('.','',1).isnumeric()) and \
word[0]!="'s" ]
token_dist = FreqDist(stop_tokens)
print("\nCorpus contains", len(token_dist.items()), \
" unique terms after removing stop words.\n")
for word, frequency in token_dist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word[0], frequency))

# Lemmatization - Stemming with POS
# WordNet Lematization Stems using POS
stemmer = SnowballStemmer("english")
wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
wnl = WordNetLemmatizer()
stemmed_tokens = []
for token in stop_tokens:
    term = token[0]
    pos = token[1]
    pos = pos[0]
    try:
        pos = wn_tags[pos]
        stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
    except:
        stemmed_tokens.append(stemmer.stem(term))
# Get token distribution
fdist = FreqDist(stemmed_tokens)
print("\nCorpus contains", len(fdist.items())," unique terms after Stemming.\n")

#Printing Top 20 terms
print('Top 20 terms sorted by frequency')
for word, freq in fdist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word, freq))
fdist_top = nltk.probability.FreqDist()
for word, freq in fdist.most_common(20):
    fdist_top[word] = freq
fdist_top.plot()