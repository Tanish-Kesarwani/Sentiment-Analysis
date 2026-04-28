from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import pandas as pd
import json

from models.pipeline import get_sentiment

app = FastAPI(title="Sentiment Analysis API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can replace * with frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("processed_data.csv")
except Exception as e:
    df = None
    print("Error loading processed_data.csv:", e)

# ---------------- LOAD ANALYTICS ----------------
try:
    with open("analytics.json") as f:
        analytics = json.load(f)
except Exception as e:
    analytics = None
    print("Error loading analytics.json:", e)

# ---------------- REQUEST MODEL ----------------
class TextRequest(BaseModel):
    text: str


# ---------------- ROOT ----------------
@app.get("/")
def home():
    return {"message": "API running 🚀"}


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    sentiment = get_sentiment(text)

    return {
        "text": text,
        "sentiment": sentiment
    }


# ---------------- SENTIMENT DISTRIBUTION ----------------
@app.get("/sentiment-distribution")
def sentiment_distribution():
    if analytics is None:
        raise HTTPException(status_code=500, detail="Analytics not loaded")

    return analytics["sentiment_distribution"]


# ---------------- TOPIC DISTRIBUTION ----------------
@app.get("/topic-distribution")
def topic_distribution():
    if analytics is None:
        raise HTTPException(status_code=500, detail="Analytics not loaded")

    return analytics["topic_distribution"]


# ---------------- TOPIC SENTIMENT ----------------
@app.get("/topic-sentiment")
def topic_sentiment():
    if analytics is None:
        raise HTTPException(status_code=500, detail="Analytics not loaded")

    return analytics["topic_sentiment"]


# ---------------- SEARCH ----------------
@app.get("/search")
def search(keyword: str):
    try:
        if df is None:
            return {"error": "Data not loaded"}

        keyword = keyword.strip()

        filtered = df[
            df['processed_text']
            .fillna("")
            .astype(str)
            .str.contains(keyword, case=False, na=False, regex=False)
        ]

        return filtered.head(50).to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


# ---------------- TOP WORDS ----------------
from sklearn.feature_extraction.text import TfidfVectorizer

@app.get("/top-words")
def top_words(sentiment: str):
    try:
        if df is None:
            return {"sentiment": sentiment, "top_words": []}

        sentiment = sentiment.strip().lower()

        temp = df.copy()

        temp["sentiment"] = temp["sentiment"].astype(str).str.lower()
        temp["processed_text"] = temp["processed_text"].fillna("").astype(str)

        texts = temp[temp["sentiment"] == sentiment]["processed_text"]

        texts = texts[texts.str.strip() != ""]

        if len(texts) == 0:
            return {"sentiment": sentiment, "top_words": []}

        vec = TfidfVectorizer(stop_words="english", max_features=20)
        X = vec.fit_transform(texts)

        words = vec.get_feature_names_out()
        scores = X.sum(axis=0).A1

        top = [words[i] for i in scores.argsort()[::-1][:10]]

        return {
            "sentiment": sentiment,
            "top_words": top
        }

    except Exception as e:
        return {
            "sentiment": sentiment,
            "top_words": [],
            "error": str(e)
        }