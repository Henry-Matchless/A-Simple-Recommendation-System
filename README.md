# A-Simple-Recommendation-System
A Simple Recommendation System

1. Extract data-film title, film type, film director, film actor and film plot 
2. Cleaning data- 
        The movie story uses rake_nltk to remove the stop words and sort the keywords. 
        Film directors and film actors remove spaces and take last name and first name as one word 
3. Splice all keywords into bag_of_words and calculate the similarity. 
4. Top10 recommendation for designated movies. 
Main technical: rate_nltk, cosine_similarity of sklean, CountVectorizer in skean
