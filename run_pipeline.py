# import pandas as pd
# import re
# import json
# import nltk
# from nltk.corpus import stopwords
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation

# # Download stopwords (only first time)
# nltk.download('stopwords')

# # ---------------- LOAD DATA ----------------
# df = pd.read_csv(r"C:\Users\deves\OneDrive\Desktop\sentiments\Sentiment-Analysis\Twitter_Data.csv")

# # ---------------- PREPROCESSING ----------------
# stop_words = set(stopwords.words('english'))

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^a-z\s]', '', text)
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return " ".join(words)

# # If your column is named differently, adjust here
# df['processed_text'] = df['clean_text'].apply(clean_text)

# # ---------------- SENTIMENT ----------------
# analyzer = SentimentIntensityAnalyzer()

# def get_sentiment(text):
#     score = analyzer.polarity_scores(text)['compound']
#     if score > 0.05:
#         return "positive"
#     elif score < -0.05:
#         return "negative"
#     else:
#         return "neutral"

# df['sentiment'] = df['processed_text'].apply(get_sentiment)

# # ---------------- TF-IDF ----------------
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['processed_text'])

# # ---------------- LDA ----------------
# lda = LatentDirichletAllocation(n_components=4, random_state=42)
# lda.fit(X)

# topic_values = lda.transform(X)
# df['topic'] = topic_values.argmax(axis=1)

# # ---------------- MAP TOPICS ----------------
# topic_map = {
#     0: "National Achievements / Defense",
#     1: "Political Campaign / Leadership",
#     2: "Elections & Party Politics",
#     3: "Economic & Public Issues"
# }

# df['topic_name'] = df['topic'].map(topic_map)

# # ---------------- SAVE PROCESSED DATA ----------------
# df.to_csv("processed_data.csv", index=False)

# # ---------------- ANALYTICS ----------------
# sentiment_dist = df['sentiment'].value_counts().to_dict()
# topic_dist = df['topic_name'].value_counts().to_dict()

# topic_sentiment = (
#     pd.crosstab(df['topic_name'], df['sentiment'], normalize='index') * 100
# ).round(2).to_dict()

# # Save analytics
# with open("analytics.json", "w") as f:
#     json.dump({
#         "sentiment_distribution": sentiment_dist,
#         "topic_distribution": topic_dist,
#         "topic_sentiment": topic_sentiment
#     }, f)

# print("✅ Pipeline complete. processed_data.csv and analytics.json created.")
import pandas as pd
import re
import json
import nltk
import torch
from nltk.corpus import stopwords
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------------- DOWNLOAD NLTK ----------------
nltk.download("stopwords")

# ---------------- GPU CHECK ----------------
print("Checking GPU...\n")

print("Torch Version:", torch.__version__)

if torch.cuda.is_available():
    print("✅ CUDA Available")
    print("✅ CUDA Version:", torch.version.cuda)
    print("✅ GPU Name:", torch.cuda.get_device_name(0))
    print("✅ GPU Count:", torch.cuda.device_count())
    print("✅ Current Device:", torch.cuda.current_device())
    print("✅ Using GPU for BERT\n")

    device = 0
    batch_size = 64

else:
    print("❌ GPU Not Found")
    print("⚠️ Using CPU Instead\n")

    device = -1
    batch_size = 16

# ---------------- LOAD DATA ----------------
df = pd.read_csv(
    r"C:\Users\deves\OneDrive\Desktop\sentiments\Sentiment-Analysis\Twitter_Data.csv"
)

print("✅ CSV Loaded")
print("Rows Found:", len(df), "\n")

# ---------------- PREPROCESSING ----------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Detect source column automatically
if "clean_text" in df.columns:
    df["processed_text"] = df["clean_text"].apply(clean_text)

elif "text" in df.columns:
    df["processed_text"] = df["text"].apply(clean_text)

else:
    first_col = df.columns[0]
    df["processed_text"] = df[first_col].apply(clean_text)

print("✅ Text Cleaning Done\n")

# ---------------- LOAD MODEL ----------------
print("Loading BERT Model...\n")

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    device=device,
    use_safetensors=True
)

def map_label(label):
    if label == "LABEL_2":
        return "positive"
    elif label == "LABEL_0":
        return "negative"
    else:
        return "neutral"

# ---------------- SENTIMENT ANALYSIS ----------------
print("Running Sentiment Analysis...\n")

texts = df["processed_text"].fillna("").tolist()
all_sentiments = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]

    try:
        results = classifier(
            batch,
            truncation=True,
            max_length=128,
            batch_size=batch_size
        )

        for res in results:
            all_sentiments.append(map_label(res["label"]))

        print(f"Processed {i + len(batch)} / {len(texts)}")

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"GPU Used: {used:.2f} MB | Reserved: {reserved:.2f} MB")

    except Exception as e:
        print("Batch Error:", e)

        for _ in batch:
            all_sentiments.append("neutral")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df["sentiment"] = all_sentiments

print("\n✅ Sentiment Analysis Completed\n")

# ---------------- TF-IDF ----------------
print("Running TF-IDF...\n")

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["processed_text"])

print("✅ TF-IDF Completed\n")

# ---------------- LDA ----------------
print("Running Topic Modeling...\n")

lda = LatentDirichletAllocation(
    n_components=4,
    random_state=42
)

lda.fit(X)

topic_values = lda.transform(X)
df["topic"] = topic_values.argmax(axis=1)

# ---------------- TOPIC LABELS ----------------
topic_map = {
    0: "National Achievements / Defense",
    1: "Political Campaign / Leadership",
    2: "Elections & Party Politics",
    3: "Economic & Public Issues"
}

df["topic_name"] = df["topic"].map(topic_map)

print("✅ Topic Modeling Completed\n")

# ---------------- SAVE OUTPUT ----------------
df.to_csv("processed_data.csv", index=False)
print("✅ processed_data.csv Saved")

# ---------------- ANALYTICS ----------------

sentiment_dist = df["sentiment"].value_counts().to_dict()
topic_dist = df["topic_name"].value_counts().to_dict()

topic_sentiment = (
    pd.crosstab(
        df["topic_name"],
        df["sentiment"],
        normalize="index"
    ) * 100
).round(2).to_dict()

analytics = {
    "sentiment_distribution": sentiment_dist,
    "topic_distribution": topic_dist,
    "topic_sentiment": topic_sentiment
}

with open("analytics.json", "w") as f:
    json.dump(analytics, f, indent=4)

print("✅ analytics.json Saved")

# ---------------- FINAL GPU REPORT ----------------
if torch.cuda.is_available():
    print("\n🎯 Final GPU Report")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Build:", torch.version.cuda)
    print(
        "Memory Reserved:",
        round(torch.cuda.memory_reserved(0) / 1024**2, 2),
        "MB"
    )

print("\n🚀 Full Pipeline Complete")