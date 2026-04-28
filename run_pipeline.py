import pandas as pd
import re
import json
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download stopwords (only first time)
nltk.download('stopwords')

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"C:\Users\Hp\Desktop\sentiment\Twitter-and-Reddit-Sentimental-analysis\Twitter_Data.csv")

# ---------------- PREPROCESSING ----------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# If your column is named differently, adjust here
df['processed_text'] = df['clean_text'].apply(clean_text)

# ---------------- SENTIMENT ----------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df['sentiment'] = df['processed_text'].apply(get_sentiment)

# ---------------- TF-IDF ----------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_text'])

# ---------------- LDA ----------------
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X)

topic_values = lda.transform(X)
df['topic'] = topic_values.argmax(axis=1)

# ---------------- MAP TOPICS ----------------
topic_map = {
    0: "National Achievements / Defense",
    1: "Political Campaign / Leadership",
    2: "Elections & Party Politics",
    3: "Economic & Public Issues"
}

df['topic_name'] = df['topic'].map(topic_map)

# ---------------- SAVE PROCESSED DATA ----------------
df.to_csv("processed_data.csv", index=False)

# ---------------- ANALYTICS ----------------
sentiment_dist = df['sentiment'].value_counts().to_dict()
topic_dist = df['topic_name'].value_counts().to_dict()

topic_sentiment = (
    pd.crosstab(df['topic_name'], df['sentiment'], normalize='index') * 100
).round(2).to_dict()

# Save analytics
with open("analytics.json", "w") as f:
    json.dump({
        "sentiment_distribution": sentiment_dist,
        "topic_distribution": topic_dist,
        "topic_sentiment": topic_sentiment
    }, f)

print("✅ Pipeline complete. processed_data.csv and analytics.json created.")