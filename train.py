import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 🔥 Bigger & smarter dataset
data = {
    "text": [
        # Unrealistic
        "I will become rich overnight",
        "Earn 1 lakh per day easily",
        "Get success without effort",
        "Become expert instantly",
        "No work and earn money",
        "Make money fast without skills",
        "Earn money without doing anything",
        "Get rich without working",
        "No effort but high income",
        "Instant success without hard work",
        "Success in 1 day",
        "Quick money without skills",

        # Realistic
        "Learn AI in 2 years with practice",
        "Crack job with daily study",
        "Build skills step by step",
        "Consistent learning leads to success",
        "Practice daily to improve skills",
        "Hard work leads to success",
        "Improve skills with regular effort",
        "Success needs time and dedication",
        "Learning daily brings growth",
        "Work hard and earn money",
        "Practice coding every day",
        "Effort leads to improvement"
    ],
    "label": [2]*12 + [0]*12
}

df = pd.DataFrame(data)

# 🔥 BIG CHANGE: ngrams (context samjhega)
vectorizer = TfidfVectorizer(ngram_range=(1,2))

X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print(" Model trained 🚀")