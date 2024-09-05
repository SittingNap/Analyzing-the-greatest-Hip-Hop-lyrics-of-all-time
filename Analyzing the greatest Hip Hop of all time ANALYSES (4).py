#!/usr/bin/env python
# coding: utf-8

# # Big Data and Automated Content Analysis Final Project
# 
# # Analysing the lyrics of the greatest hip hop artists of all time
# 
# Lukas Pechacek
# 
# 11954922
# 
# 03/06/2022
# 
# Damian Trilling
# 
# University of Amsterdam

# In[1]:


# General packages and dictionary analysis
import os
import numpy as np
import json
import pprint
import glob
import tarfile
import bz2
import urllib.request
import re
import pickle
import csv
import requests
import pandas as pd
from collections import Counter

from nltk.tokenize import TreebankWordTokenizer
import matplotlib.pyplot as plt
import matplotlib.style as style 
import seaborn as sns

# Topic Modeling
import gensim
from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import (
    CoherenceModel)
from gensim import corpora, models
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD #LAD algorithm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #vectorize the lyrics
from sklearn.model_selection import GridSearchCV #hyperparameter tuning
import pyLDAvis 
import pyLDAvis.lda_model
import pyLDAvis
#vis
import pyLDAvis.gensim_models as gensimvis
import logging
import matplotlib.colors as mcolors
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

#spacy
import spacy
from nltk.corpus import stopwords


# Tokenization
import nltk
from nltk.tokenize import (TreebankWordTokenizer, 
                           WhitespaceTokenizer)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer)
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer


# For plotting word clouds
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn import preprocessing


# Natural language processing
import spacy

import ufal.udpipe
from gensim.models import KeyedVectors, Phrases
from gensim.models.phrases import Phraser
from ufal.udpipe import Model, Pipeline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
nltk.download('wordnet')


# # Data collection
# 
# 
# The list of artists was retrieved from the website ranker.com. The BeautifulSoup package was used to scrape the top 25 hip hop artists as well as the top 25 hip hop bands. The two lists were then joined together and saved as a text file. Afterwards, the Genius API was used to get the top 90 songs by each artist and the corresponding URLs to the song lyrics. Some artists did not have 90 songs on Genius. A list of these three artists was made and 60 songs per artist on this list were still retrieved. Once again, the BeautifulSoup package was used to scrape the lyrics from the retrieved URLs. Artists that had less than 30 songs on Genius were dropped from the list. Ultimately, 44 artists with 3870 songs were retained for further analysis. After scraping the lyrics all the information, namely the artists name, the song URL for each song and lyrics of each song were saved into a csv file. 
# 

# # Exploratory Analysis and preprocessing

# Here are my exploratory analyses and preprocessing steps. Please find a more comprehensive summary of these before the topic modelling section.

# In[2]:


#importing hip hop artists and lyrics dataframe
hiphopDF = pd.read_csv('hiphopDF.csv')
print(hiphopDF.shape)
hiphopDF.head()


# In[3]:


pd. set_option('display.max_rows', None) #to get better overview


# In[4]:


hiphopDF


# There are 3870 songs with the attributes of 'Artist', 'lyrics' and the song 'url'. First task will be to analyse the data set and get rid of inconsistencies. By inspecting the tail of the data frame it is apparent that there are already some missing values. Some of the first cleaning task include:
# 
# - drop missing values
# - check dublicates 
# - drop duplicates

# In[5]:


#Checking for missing values
hiphopDF.isna().sum()


# There are 86 songs with missing lyrics. It does not make sense to keep these for further analyses. Therefore these rows will be dropped.

# In[6]:


#Remooving missing values
hiphopDF.dropna(inplace = True)
hiphopDF.reset_index(inplace = True)


# In[7]:


hiphopDF.isna().sum()


# In[8]:


#checking shape of dataframe
hiphopDF.shape


# In[9]:


hiphopDF.head()


# Inspecting sample lyrics:
# seems like the songs do not have paragraphs but multiple lines as sentences.

# In[10]:


#inspecting the lyrics
hiphopDF.loc[0, 'lyrics']


# In[11]:


#Checking how many songs with lyrics we have after removing missing values
hiphopDF['Artist'].value_counts()


# Next, duplicates will be dropped based on lyrics column. The rows are first checked for duplicates and put into another dataframe.

# In[12]:


df = hiphopDF[hiphopDF.duplicated(['Artist', 'lyrics'])]
df.shape


# In[13]:


#Printing dataframe with duplicates
df


# We see 18 rows of duplicates. As such, the original hiphop dataframe will have 18 rows less.

# In[14]:


#Dropping duplicates
hiphopDF.drop_duplicates(['Artist', 'lyrics'], inplace=True)


# In[15]:


hiphopDF.shape


# Next, the lyircs themselves will be examined more closely. There may a lot of instrumental songs as Hip Hop artists often put those into their albums. Otherwise it could also be that some songs may not have been scraped correctly, and as such their value remains questionable. It thus makes sense to :
# - Create a new column based on word count
# - drop those rows that have a non sufficient word count
# 
# Afterwards:
# 
# - removal of stop words, punctuation. 
#     - might have to extend stop words list to include words in lyrics such as 'la', 'ha' etc
# - explore artists with most amount of words
# - explore artists with most amount of unique words
# - lemmatization

# https://stackoverflow.com/questions/50821143/pandas-dataframe-count-unique-words-in-a-column-and-return-count-in-another-col

# In[16]:


#create a column on the dataframe showing the total number of words for each song
hiphopDF['Word Count'] = hiphopDF['lyrics'].str.split().str.len()## Split list into new series. 

hiphopDF


# In[17]:


hiphopDF['Word Count'].describe()


# Unfortunately, it seems like a lot of songs have not been scraped correctly and did not return the desired output. From personal knowledge there are indeed some instrumentals. However, also from personal knowledge and inspecting the lyrics more closely, something indeed did not get scraped correctly (I tried the process again but got the same output). For now, songs with less than 25 words will be dropped. I still believe that those with more than 25 can still add value to the analysis.

# In[18]:


#using loc method to locate all songs with more than 0 and less than 25 lyrics
indexsongs = hiphopDF.loc[(hiphopDF['Word Count']>0) & (hiphopDF['Word Count'] < 25)].index
hiphopDF.drop(indexsongs , inplace=True)


# In[19]:


len(hiphopDF)


# In[20]:


#Creating dataframe grouped by lyrics and artitst, and checking values counts as well we sorting them by size.
songsDF = pd.DataFrame(hiphopDF.groupby('Artist')['lyrics'].count().sort_values())
songsDF


# We can see that Ghostface Killah has the most songs, while Eric B. and Rakim have the least amount of songs returned.

# - stopword removal
# 
# I believe it makes sense to add more words to stop words list that often appear in the lyrics, such as 'uh' 'oh', for instance.

# In[21]:


stop_words = stopwords.words('english')
stop_words.extend(('la','la la la la', 'lalala', 're', 'oh oh oh', 'like', 'yeah ah', 'ah yeah', 'laaaa', 'oh', 's', 'uh', 'ah', 'ooooh', 'ohhh', 'yo', 'cha', 
                       'em', 'duh', 'im', 'yeah yeah', 'yeah', 'nigga', 'niggas', 'ai', 'thats', 'ta', 'na', 'pussy', 'dick',  'motherfucker', 'htotheatotheitothettothei',
                  'ge', 'effizzect', 'ladidadada', 'ai', 's', 'nananananana', 'una', 'un', 'el', 'de', 'que', 'get', 'got', 'say', 
                  'do', 'let', 'come', 'tha'))


# In[22]:


print(stop_words)


# Function removing stopwords and punctuation as well as spaces that can occur at the start or end of an article. This function was taken from https://www.youtube.com/watch?v=dtK7Xhn8XjY&t=129s&ab_channel=SolveBusinessProblemsUsingAnalytics and adapted. I do not claim this function as my own.

# In[23]:


def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip() 
    text = re.sub(' +', ' ', text) 
    text = re.sub(r'[^\w\s]','',text) 
    
    
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence)
    return text


# Creating a new column with cleaned lyrics

# In[24]:


hiphopDF['clean_lyrics'] = hiphopDF['lyrics'].apply(process_text)


# Seems like it worked. But there are still words visible such as'im', that should be removed. Therefore, another column is created, based on stop word removal from the already clean_lyrics column.

# In[25]:


hiphopDF.head()


# In[26]:


#Removing more stop words
hiphopDF['clean_lyrics1'] = hiphopDF['clean_lyrics'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# Let's see if it did anything

# In[27]:


hiphopDF.head()


# In[28]:


#Creating a second column with clean lyrics word count
hiphopDF['clean_word_count1'] = hiphopDF['clean_lyrics1'].str.split().str.len()
hiphopDF


# It seems like removing stop words on the already cleaned lyrics removed further words, such as 'im', visible on row[0]. So it seems for now the clean_lyrics1 column is more suitable.

# In[29]:


#trying to remove non english words
hiphopDF[hiphopDF['clean_lyrics1'].map(lambda x: x.isascii())]


# In[30]:


hiphopDF.groupby('Artist')['Word Count'].sum().sort_values()


# In[31]:


#Word coutn per artist before stop word removal
plt.figure(figsize=[8,20])
sns.barplot(x=hiphopDF['Word Count'], y=hiphopDF['Artist'], palette = 'magma')
plt.show()


# Let's see which artists have the most word counts after the stop word removal.

# In[32]:


hiphopDF.groupby('Artist')['clean_word_count1'].sum().sort_values()


# In[33]:


plt.figure(figsize=[8,20])
sns.barplot(x=hiphopDF['clean_word_count1'], y=hiphopDF['Artist'], palette = 'magma')
plt.show()


# In[34]:


#Making a dataframe in order to extract the number of unique words per artist
unique_wordsDF = pd.DataFrame(hiphopDF.groupby('Artist')['clean_lyrics1'].sum().sort_values())
unique_wordsDF.reset_index(inplace=True) 
unique_wordsDF.tail()


# In[35]:


len(unique_wordsDF.iat[42, 1])


# In[36]:


#Creating a set of unique words
unique_words = set()
unique_wordsDF['clean_lyrics1'].str.lower().str.split().apply(unique_words.update)
print(unique_words)


# In[37]:


unique_wordsDF['n_unique'] = [len(set(x.split())) for x in unique_wordsDF['clean_lyrics1'].str.lower()]
#for each value in the column, split the string into words and keep the unique value values in set and get the lenght of the set.


# In[38]:


unique_wordsDF.groupby('Artist')['n_unique'].sum().sort_values()


# In[39]:


unique_wordsDF.head()


# In[40]:


text = hiphopDF['clean_lyrics1'].values 

wordcloud = WordCloud().generate(str(text))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[41]:


##https://www.linkedin.com/pulse/new-way-look-artist-from-lyrics-wordclouds-margot-lepizzera/
#to get a better overview for further preprocessing, the column will be turned into a string
all_lyrics = ""
for index, row in hiphopDF.iterrows():
    lyrics_decoded = row['clean_lyrics1'].lower()
    lyrics_nobacklash = lyrics_decoded.replace("\"", "")
    all_lyrics = all_lyrics + " " + lyrics_nobacklash


# In[42]:


all_lyrics


# In[43]:


#https://www.linkedin.com/pulse/new-way-look-artist-from-lyrics-wordclouds-margot-lepizzera/
#identifying 10 most common words

#tokenizing

tok = nltk.tokenize.word_tokenize(all_lyrics)

counter = Counter(tok)
counter.most_common(10)


# It seems like there are still a lot of words that do not indicate many topics. Further preprocessing is needed before starting the analysis. I will also append the stop word list with some of these words such as 'get'and 'got'.

# In[44]:


tokens = nltk.tokenize.word_tokenize(all_lyrics)
print(tokens)


# In[45]:


len(tokens)


# lemmatization, code inspired by - https://www.youtube.com/watch?v=TKjjlp5_r7o&ab_channel=PythonTutorialsforDigitalHumanities

# In[46]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #words to be included
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #disabling parser
    texts_out = [] #creating empty list
    for text in texts: #for each element in the text
        doc = nlp(text) #creating doc object
        new_text = [] #new text list
        for token in doc: #for each token in list
            if token.pos_ in allowed_postags: #if the token in verb, adjective, noun or adverb, append to new_text list
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


# In[47]:


lemm_lyrics = hiphopDF['clean_lyrics1'].values
data_lemma = lemmatization(lemm_lyrics, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[48]:


print('First item of the lemmatized lyrics by Spacy')
print(data_lemma[0])


# In[49]:


#converting strings to list
lyrics_list = [song.split() for song in data_lemma] 


# In[50]:


print(lyrics_list)


# # To sum up: Exploratory Analyses and preprocessing
# 
# Exploratory analyses revealed that before the stop word removal Kendrick Lamar was the artist with the highest word count, while NWA was the artist with the lowest word count out of the artists of which 90 songs could be extracted. Interestingly, after stop word removal, Ghostface Killa became the artist with highest word count. Ghostface Killa and The Fugees were the artists with the highest number of unique words in the lyrics while rapper 50 Cent was the artist with the least number of unique words. 
# 
# The csv file was then turned into a Pandas data frame, which consisted of 3870 rows and 3 columns. After checking for missing values, it became clear that the lyrics of 86 songs were not scraped at all. It made no sense to keep those in the dataset and as such were dropped. Next, it was checked whether some songs were duplicated. After inspection 18 rows were dropped as duplicates. A column was then created based on the word count per song. This was to see whether there were some songs with a suspiciously little number of words. This could be due to some songs not having been scraped correctly or some songs only being the instrumental version, and as such not very useful for the analysis. All songs with a word count between 0 and 25 were ultimately dropped from the data set. Anything above a word count of 25 still seemed as if it could add to the analysis. Consequently, the final data set used for the analysis consisted of 3433 songs. The next step of the pre-processing was to remove stopwords from the lyrics. Before this was done the stop word list was appended with a few more items after some inspection of the lyrics. Lyrics such as ‘la’, ‘lalala’, ‘oh’, ‘yeah yeah’ among others were added to the list as these were deemed as unnecessary words for the analysis. Lastly, the lyrics were lemmatized using the spacy lemmatizer to group together inflected forms of a word so they can be analysed as a single item, essentially retaining more words while removing different forms of the same word. Nouns, adjectives, verbs and adverbs were kept for kept for the analysis. 

# # Topic Modelling

# To model simultaneously which topics are found in the whole corpus of song lyrics, a topic modelling approach with latent Dirichlet Allocation (LDA) was chosen. This will allow us to see which topics are present in which documents, whilst also being able to make connections between words even if they are not in the same document (Maier et al., 2018). To start, the analysis initiated with the use of the count vectorizer conducting iterations on up to 100 topics. Next, the TfIdf vectorizer was used to perform the operation on the same number of topics. Afterwards, several operations were performed with different parameter configurations for the alpha hyperparameter. To see the results of these configurations a line plot was constructed indicating the perplexity and coherence scores. Perplexity being a score of how well the model is able to predict the word distribution while coherence is a score of how semantically coherent the topics are. (Mimno et al., 2011). A visualization by the genism package was also constructed on all models in order to get a better overview of the most relevant terms per topic. These scores indicated that for all models the inflection point was at k=20 topics. Lastly, a model was created taking into account bigrams and trigrams in order to measure words that could be used with each other. 

# I will first determine the number of topics I wish to explore. Afterwards, I will iterate over the lemmatized texts and fit the models accordingly. I will start with a model suing the Count Vectorizer, inspect the results as well as visualizations. Next, I will run the same model only using the TfIdf vectorizer. Afterwards I will perform some hyperparameter tuning and also construct a text made out of bigrams and trigrams and see whether these operations improve the models.
# 
# Lastly, I will discuss all the results.

# In[51]:


#Creating a list of numbers of topics
NTOPICS = [3, 5, 7, 10, 15, 20, 25, 30, 50, 100]


# In[52]:


for k in NTOPICS:
    #Creating BOW representations of the text
    id2word = corpora.Dictionary(k for k in lyrics_list)  #input into LDA
    mm =[id2word.doc2bow(text) for text in lyrics_list] #input into LDA
    # Train the LDA models. 
    lda_model = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics= k, alpha="auto") 
    # Print the topics. 
    print ("\nModel with ",k," topics results in the following words:\n\n")
    for top in lda_model.print_topics(num_topics= k, num_words=5): 
        print ("\n",top)


# Now the visualization

# In[1]:


vis_data = gensimvis.prepare(lda_model, mm, id2word)
pyLDAvis.display(vis_data)


# From running the first topic model we can see that there are no meaningful conclusions to be drawn yet. From the visualization we can see that there are a lot of overlapping topics, with the most relevant terms also not being a good indication for what these topics could be about. 

# In[54]:


result = []
for k in [5, 10, 15, 20, 25, 30]:
  m = LdaModel(mm,num_topics=k,id2word=id2word, 
          random_state=123, alpha="auto")
  perplexity = m.log_perplexity(mm)
  coherence=CoherenceModel(model=m,corpus=mm,
      coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# From the perplexity and coherence plot we also can see a clear inflection point at k=20. However, after inspecting the visualization it seems better to try our hand using the tfidf vectorizer. By using the tdidf vectorizer we are transforming our text into a vector that on the basis of how many times the word occurs in the documents.

# In[55]:


for k in NTOPICS:
    id2word_m2 = id2word
    ldacorpus_m2 = mm       
    tfidfcorpus_m2 = models.TfidfModel(ldacorpus_m2)
    mylda_m2 = models.ldamodel.LdaModel(corpus=tfidfcorpus_m2[ldacorpus_m2],id2word=id2word_m2,num_topics=k, passes=10)
    print ("\nModel with ",k," topics results in the following words:\n\n")
    print (mylda_m2.print_topics(num_words=5))


# In[2]:


vis_data1 = gensimvis.prepare(mylda_m2, ldacorpus_m2, id2word_m2, mds= 'mmds')
pyLDAvis.display(vis_data1)


# From the visualiztion we can see that there are now more non overlapping topics, which is great! Unfortunately, the topics still do not seem to be clear cut. Let's try tuning the parameters.

# Let's plot the coherence and perplexity scores for this model. I will set the number of topics to 15 and the change up the alpha parameter to see how this affects our inflection point. 

# In[57]:


result = []
for k in [5, 10, 15, 20, 25, 30, 35, 40, 50, 100]:
  m = LdaModel(ldacorpus_m2, num_topics=k, id2word=id2word_m2,
  random_state=123, alpha="asymmetric")
  perplexity = m.log_perplexity(ldacorpus_m2)
  coherence=CoherenceModel(model=m,corpus=ldacorpus_m2,
  coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# In[58]:


result = []
for k in [5, 10, 15, 20, 25, 30, 35, 40, 50, 100]:
  m = LdaModel(ldacorpus_m2, num_topics=k, id2word=id2word_m2,
  random_state=123, alpha="auto")
  perplexity = m.log_perplexity(ldacorpus_m2)
  coherence=CoherenceModel(model=m,corpus=ldacorpus_m2,
  coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# In[59]:


result = []
for k in [5, 10, 15, 20, 25, 30, 35, 40, 50, 100]:
  m = LdaModel(ldacorpus_m2, num_topics=k, id2word=id2word_m2,
  random_state=123, alpha="symmetric")
  perplexity = m.log_perplexity(ldacorpus_m2)
  coherence=CoherenceModel(model=m,corpus=ldacorpus_m2,
  coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# Changing the alpha parameter did not change the outcome by a lot. Let's therefore try running a model with 15 topics and the alpha set to auto.

# In[60]:


Topics20 = 20


# In[61]:


id2word_m2T = id2word
ldacorpus_m2T = mm       
tfidfcorpus_m2T = models.TfidfModel(ldacorpus_m2T)
mylda_m2T = models.ldamodel.LdaModel(corpus=tfidfcorpus_m2T[ldacorpus_m2T],id2word=id2word_m2T,num_topics=Topics20, passes=10, alpha='auto')
print ("\nTuned Model with 20 topics results in the following words:\n\n")
print (mylda_m2T.print_topics(num_words=10))


# Running the model does not seem to give us a good indication of topics. Topic 5 could be about the most mentioned areas in rap songs, namely bronx in New York and compton in south central Los Angeles with the words 'brookly', 'bronx', 'compton' and 'south'.

# In[62]:


vis_data1T = gensimvis.prepare(mylda_m2T, ldacorpus_m2T, id2word_m2T, mds= 'mmds')
pyLDAvis.display(vis_data1T)


# Next, let's try making some bigrams. 

# In[63]:


#https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo_bigrams_trigrams.ipynb
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(data_lemma)

print (data_words[0][0:20])


# In[64]:


# Create a dictionary representation of the documents.
dictionary = Dictionary(data_words)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)


# In[65]:


#https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo_bigrams_trigrams.ipynb
#Bigrams and trigrams
bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return([bigram[doc] for doc in texts])

def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)



# In[66]:


print ('Model 3: Controlling for bigrams and trigrams)')
id2word_m3 = corpora.Dictionary(data_bigrams_trigrams)                       
ldacorpus_m3 = [id2word_m3.doc2bow(doc) for doc in data_bigrams_trigrams]
tfidfcorpus_m3 = models.TfidfModel(ldacorpus_m3)
lda_m3 = models.ldamodel.LdaModel(corpus=tfidfcorpus_m3[ldacorpus_m3],id2word=id2word_m3,num_topics=30, passes = 10, alpha='auto', eta='auto')
lda_m3.print_topics(num_words=10)


# In[67]:


vis_data2 = gensimvis.prepare(lda_m3, ldacorpus_m3, id2word_m3, mds= 'mmds')
pyLDAvis.display(vis_data2)


# In[68]:


result = []
for k in [5, 10, 15, 20, 25, 30]:
  m = LdaModel(ldacorpus_m3, num_topics=k, id2word=id2word_m3,
  random_state=123, alpha="symmetric")
  perplexity = m.log_perplexity(ldacorpus_m3)
  coherence=CoherenceModel(model=m,corpus=ldacorpus_m3,
  coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# In[69]:


result = []
for k in [5, 10, 15, 20, 25, 30]:
  m = LdaModel(ldacorpus_m3, num_topics=k, id2word=id2word_m3,
  random_state=123, alpha="auto")
  perplexity = m.log_perplexity(ldacorpus_m3)
  coherence=CoherenceModel(model=m,corpus=ldacorpus_m3,
  coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# In[70]:


result = []
for k in [5, 10, 15, 20, 25, 30]:
  m = LdaModel(ldacorpus_m3, num_topics=k, id2word=id2word_m3,
  random_state=123, alpha="asymmetric")
  perplexity = m.log_perplexity(ldacorpus_m3)
  coherence=CoherenceModel(model=m,corpus=ldacorpus_m3,
  coherence="u_mass").get_coherence()
  result.append(dict(k=k, perplexity=perplexity, 
                     coherence=coherence))

result = pd.DataFrame(result)
result.plot(x="k", y=["perplexity", "coherence"])
plt.show()


# In[73]:


print ('Tuned Model 3: Controlling for bigrams and trigrams)')
id2word_m3T = corpora.Dictionary(data_bigrams_trigrams)                       
ldacorpus_m3T = [id2word_m3T.doc2bow(doc) for doc in data_bigrams_trigrams]
tfidfcorpus_m3T = models.TfidfModel(ldacorpus_m3T)
lda_m3T = models.ldamodel.LdaModel(corpus=tfidfcorpus_m3T[ldacorpus_m3T],id2word=id2word_m3T,num_topics=15, alpha='auto', eta='auto')
lda_m3T.print_topics(num_words=5)


# In[72]:


vis_data2T = gensimvis.prepare(lda_m3T, ldacorpus_m3T, id2word_m3T, mds= 'mmds')
pyLDAvis.display(vis_data2T)


# # Results 

# Running the LDA topic model on up to 100 topics with the count vectorizer did not provide this analysis with any very good results. In fact, most of the words were not particularly good at harmonizing any topics. There was also a lot of overlap between the topics. Before running further topic modelling analysis. I went back to further customize the stop word list to exclude words that do not give much meaning in order to create a model with words that harmonize given topic more. Running the LDA model with the TfIdf vectorizer provided some slightly more promising insights. The metrics showed the weight of all words being much less than in the model using the count vectorizer. However, a lot of the topis were still largely overlapping. From topic 15 upwards there still seemed to be a lot of overlap. After tuning the alpha hyperparameter the perplexity and coherence scores indicated the inflection point at k=20 topics. As such as model was run with the number of topics at k=20 and the alpha set to auto and the passes parameter was set to 10. After the tuning of the hyperparameter there were 2 non over lapping topics with that made sense as seen on the visualization (vis_data1T). 
# 
# Topic 2 = Topic about money, keywords: Money, dollar, bill, business, job
# 
# Topic 6 = Topic about violence, keywords: Violent, violence, ghetto
# 
# Yet, the rest of the topics did unfortunately not produce anything coherent. Next, bigrams and trigrams were introduced that somewhat successfully combined words such as ‘new’ and ‘york’ to make ‘New York’. However, by manually inspecting the metrics there not many other meaning combinations could be spotted. I still decided to plot the coherence and perplexity scores in the hope of improving the output of the model. The number of topics to be looked for ranged from 5 to 30 and with varying alpha values. After tuning these hyperparameters a final model was run on the data with bigrams and trigrams, number of topics at k=15 and the alpha and eta parameters set to auto. Unfortunately, it can be concluded here that these configurations with the data made into bigrams also does has not led to produce any coherent topics. Words on topic 3 on the untuned model on the visualization perhaps somewhat weakly loaded on the topic of ‘crime’ by containing the words ‘law’, ‘murder’ and ‘bullet’.  Topic 3 also included the word ‘New_York’ and ‘bronx’ indicating that perhaps a lot of documents in the corpus were about crime in New York. 
# 
# 

# # Sources

# Ballard, M. E., Dodson, A. R., & Bazzini, D. G. (1999). Genre of music and lyrical content: Expectation effects. The Journal of Genetic Psychology, 160(4), 476-487. 
# 
#     
# Barradas, G. T., & Sakka, L. S. (2021). When words matter: A cross-cultural perspective on lyrics and their relationship to musical emotions. Psychology of Music, 03057356211013390. 
# 
# 
# Blackman, S. (2014). Subculture theory: An historical and contemporary assessment of the concept for understanding deviance. Deviant behavior, 35(6), 496-512. 
# 
# 
# Calvert, C., Morehart, E., & Papdelias, S. (2014). Rap music and the true threats quagmire: When does one man's lyric become another's crime. Colum. JL & Arts, 38, 1. 
# 
# 
# Dunbar, A. (2019). Rap music, race, and perceptions of crime. Sociology Compass, 13(10), e12732. 
# 
# 
# Kubrin, C. E. (2005). “I see death around the corner”: Nihilism in rap music. Sociological Perspectives, 48(4), 433-459. 
# 
# 
# Lena, J. C. (2006). Social context and musical content of rap music, 1979–1995. Social Forces, 85(1), 479-495. 
# 
# 
# MacDonald, R., Kreutz, G., & Mitchell, L. (2012). What is music, health, and wellbeing and why is it important. Music, health, and wellbeing, 3-11 
# 
# 
# Maier, D., Waldherr, A., Miltner, P., Wiedemann, G., Niekler, A., Keinert, A., ... & Adam, S. (2018). Applying LDA topic modeling in communication research: Toward a valid and reliable methodology. Communication Methods and Measures, 12(2-3), 93-118. 
# 
# 
# Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011, July). Optimizing semantic coherence in topic models. In Proceedings of the 2011 conference on empirical methods in natural language processing (pp. 262-272). 
# 
# 
# North, A. C., Hargreaves, D. J., & O'Neill, S. A. (2000). The importance of music to adolescents. British journal of educational psychology, 70(2), 255-272. 
# 
# 
# Rothbaum, F., & Tsang, B. Y. P. (1998). Lovesongs in the United States and China: On the nature of romantic love. Journal of Cross-Cultural Psychology, 29(2), 306-319. 
# 
# 
# Stratton, V. N., & Zalanowski, A. H. (1994). Affective impact of music vs. lyrics. Empirical studies of the arts, 12(2), 173-184. 
# 
# 
# Tyson, E. H. (2002). Hip hop therapy: An exploratory study of a rap music intervention with at-risk and delinquent youth. Journal of Poetry Therapy, 15(3), 131-144. 
# 
# 
# van Atteveldt, W., Trilling, D., and Arc ́ıla Calder ́on, C. (2022). Computational Analysis of Communication: A Practical Introduction to the Analysis of Texts, Networks, and Images with Code Examples in Python and R. Wiley, Hoboken, NJ. 
#  
# 
