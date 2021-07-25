#!/usr/bin/Python
# -*- coding: utf-8 -*

"""""""""
1. extract data-film title, film type, film director, film actor and film plot 
2. Cleaning data- 
        The movie story uses rake_nltk to remove the stop words and sort the keywords. 
        Film directors and film actors remove spaces and take last name and first name as one word 
3. Splice all keywords into bag_of_words and calculate the similarity. 
4. top10 recommendation for designated movies. 
Main technical: rate_nltk, cosine_similarity of sklean, CountVectorizer in skean

1.提取数据---电影标题，电影类型，电影导演，电影演员，电影剧情
2.清洗数据---
      电影剧情使用rake_nltk去除停定词，对关键词排序。
      电影导演，电影演员去除空格，把姓和名作为一个单词
3.把所有的关键词拼接成bag_of_words,计算相似度。
4.对指定电影进行top10推荐。
主要的技术：rate_nltk,sklean中的cosine_similarity,skean中的CountVectorizer
"""""""""

import nltk
nltk.download('stopwords')
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', 100)
df = pd.read_csv('IMDB_Top250Engmovies2_OMDB_Detailed.csv')
print(df.head())
print(df.shape)
df = df[['Title', 'Genre', 'Director', 'Actors', 'Plot']]
df.head()
# discarding the commas between the actors' full names and getting only the first three names
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
print("====演员列表====")
print(df['Actors'][:3])

# putting the genres in a list of words
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))
print("===类型列表===")
print(df['Genre'][:3])

df['Director'] = df['Director'].map(lambda x: x.split(' '))
print("===导演列表====")
print(df['Director'][:3])

# merging together first and last name for each actor and director, so it's considered as one word
# and there is no mix up between people sharing a first name
for index, row in df.iterrows():
    # 把姓和名变成一个单词
    row['Actors'] = [x.lower().replace(' ', '') for x in row['Actors']]
    # print("变化前")
    # print(row['Director'])
    row['Director'] = ''.join(row['Director']).lower()
    # print("变化后")
    # print(row['Director'])

for index, row in df.iterrows():
    if (index < 3):
        print("===演员列表===")
        print(row['Actors'])
        print("===导演列表===")
        print(row['Director'])

# initializing the new column
df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']

    # instantiating Rake, by default is uses english stopwords from NLTK
    # and discard all puntuation characters
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)

    # getting the dictionary whith key words and their scores
    key_words_dict_scores = r.get_word_degrees()
    print("===key_words_dict_scores===")
    print(key_words_dict_scores)

    # assigning the key words to the new column
    row['Key_words'] = list(key_words_dict_scores.keys())
    print("===key_words===")
    print(row['Key_words'])

# dropping the Plot column
df.drop(columns=['Plot'], inplace=True)

df.set_index('Title', inplace=True)
print(df.head())

df['bag_of_words'] = ''
print("===df.columns====")
print(df.columns)
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col]) + ' '
        else:
            words = words + row[col] + ' '
        print("====words:====")
        print(words)
    row['bag_of_words'] = words

for index, row in df.iterrows():
    print("===bag_of_words===")
    print(row['bag_of_words'])

df.drop(columns=[col for col in df.columns if col != 'bag_of_words'], inplace=True)

print("===head:===")
print(df.head())

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
indices[:5]
print("===indices:===")
print(indices)
# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
# cosine_sim
print("===cosine_sim:===")
print(cosine_sim)


# function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim=cosine_sim):
    recommended_movies = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]
    print("===idx:====")
    print(idx)

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    print("===score_series===")
    print(score_series)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    return recommended_movies


print(recommendations('Fargo'))