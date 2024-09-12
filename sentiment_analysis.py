# Importing libraries
import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS
nltk.download('vader_lexicon')
import re

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Read the csv file into a dataframe
df = pd.read_csv("amazon_product_reviews.csv", low_memory=False)

# Extract the review texts
reviews_data = df['reviews.text']
# Removing all missing values from the feature column
df.dropna(subset=['reviews.text'], inplace=True)
# Converting to lower case
reviews_data = reviews_data.str.lower()

# Function to remove stop words
def remove_stop(x):
    return " ".join([word for word in str(x).split() if word not in STOP_WORDS])

# Updating our reviews data to not include any stop words
reviews_data = reviews_data.apply(lambda x : remove_stop(x))

# # Remove special characters, integers, stopwords, punctuation, currency and spaces
# def normalise(msg):
#     msg = re.sub('[^A-Za-z]+', ' ', str(msg))
#     doc = nlp(msg)
#     res=[]
#     for token in doc:
#         if(token.is_digit or token.is_punct or not(token.is_oov) or token.is_currency
#            or token.is_space or len(token.text) <= 2):
#             pass
#         else:
#             res.append(token.lemma_.lower())
#     return res

# reviews_data = reviews_data.apply(lambda x : normalise(x))

"""The code above is an attempt at further preprocessing the text however this would always cause my laptop
to spend a long time to present the output and turn it into an air conditioner unit, so I will emit this code
"""

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function for sentiment analysis using VADER
def analyse_sentiment(text):
    # Get the sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    # Determine the sentiment category based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Adding a column to our reviews data that determines the sentiment of each review    
reviews_data['sentiment'] = reviews_data.apply(analyse_sentiment)

# Printing the first 5 columns of our newly analysed reviews
print(reviews_data.head())
print(reviews_data['sentiment'].head())
