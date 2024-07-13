import pandas as pd

# Load the dataset
df = pd.read_csv('twitter-entity-sentiment.csv')

# Display the first few rows and inspect the columns
print(df.head())
print(df.info())

# Example of basic preprocessing (adjust as per dataset characteristics)
df = df.dropna(subset=['text'])  # Drop rows with missing text
df['text'] = df['text'].str.lower()  # Convert text to lowercase

from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to each tweet
df['sentiment'] = df['text'].apply(get_sentiment)

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.show()

from wordcloud import WordCloud

# Generate word cloud for positive sentiment tweets
positive_tweets = df[df['sentiment'] > 0]['text']
positive_text = ' '.join(positive_tweets)
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# Generate word cloud for negative sentiment tweets
negative_tweets = df[df['sentiment'] < 0]['text']
negative_text = ' '.join(negative_tweets)
wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

# Plot word clouds
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')
plt.axis('off')

plt.show()

