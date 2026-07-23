import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── 1. LOAD & LABEL ──────────────────────────────────────────────
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1
df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Shape:", df.shape)
print(df["label"].value_counts())
print("Missing values:\n", df.isnull().sum())

# ── 2. FEATURE: TEXT LENGTH ──────────────────────────────────────
df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))

# ── 3. VISUALIZATIONS ────────────────────────────────────────────

# section for the class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="label", hue="label", data=df, palette="Set2", legend=False)
plt.xticks([0, 1], ["Fake", "Real"])
plt.title("Class Distribution: Fake vs Real News")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# section for aricle length distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="text_length", hue="label", bins=50, palette="Set2")
plt.title("Article Length Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("article_length.png")
plt.show()

# for subject breakdown
plt.figure(figsize=(10, 5))
sns.countplot(y="subject", hue="label", data=df, palette="Set2")
plt.title("Article Subject by Label")
plt.tight_layout()
plt.savefig("subject_breakdown.png")
plt.show()

# word cloud matrix
fake_text = " ".join(df[df["label"] == 0]["text"].astype(str).tolist())
real_text = " ".join(df[df["label"] == 1]["text"].astype(str).tolist())

wc_fake = WordCloud(
    width=800, height=400, background_color="white", colormap="Reds", max_words=100
).generate(fake_text)
wc_real = WordCloud(
    width=800, height=400, background_color="white", colormap="Greens", max_words=100
).generate(real_text)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(wc_fake, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Most Common Words — Fake News", fontsize=14)
axes[1].imshow(wc_real, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Most Common Words — Real News", fontsize=14)
plt.tight_layout()
plt.savefig("wordclouds.png")
plt.show()

print("\nArticle length stats:")
print(df.groupby("label")["text_length"].describe())

# ── 4. PREPROCESSING ─────────────────────────────────────────────
# using the title and the text for better features
df["combined"] = df["title"] + " " + df["text"]

# cleanup; lowercase and non alphanumeric values
df["combined"] = df["combined"].str.lower()
df["combined"] = df["combined"].str.replace(r"[^a-z\s]", "", regex=True)

X = df["combined"]
y = df["label"]

# Train and test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization by converting the text o numerical values
# max_features=50000 top 50k (important)******
tfidf = TfidfVectorizer(max_features=50000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTraining samples : {X_train_tfidf.shape[0]}")
print(f"Test samples     : {X_test_tfidf.shape[0]}")
print(f"Number of features (TF-IDF): {X_train_tfidf.shape[1]}")

# ── 5. MODEL 1: LOGISTIC REGRESSION ──────────────────────────────
print("\n── Logistic Regression ──")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)

lr_acc = accuracy_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds, average="weighted")
print(f"Accuracy : {lr_acc:.4f}")
print(f"F1 Score : {lr_f1:.4f}")
print(classification_report(y_test, lr_preds, target_names=["Fake", "Real"]))

# Confusion matrix
cm_lr = confusion_matrix(y_test, lr_preds)
disp_lr = ConfusionMatrixDisplay(cm_lr, display_labels=["Fake", "Real"])
disp_lr.plot(cmap="Blues")
plt.title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig("cm_logistic.png")
plt.show()

# ── 6. MODEL 2: MULTINOMIAL NAIVE BAYES (NOTE)──────────────────────────
print("\n── Multinomial Naive Bayes ──")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_preds = nb.predict(X_test_tfidf)

nb_acc = accuracy_score(y_test, nb_preds)
nb_f1 = f1_score(y_test, nb_preds, average="weighted")
print(f"Accuracy : {nb_acc:.4f}")
print(f"F1 Score : {nb_f1:.4f}")
print(classification_report(y_test, nb_preds, target_names=["Fake", "Real"]))

# Confusion matrix
cm_nb = confusion_matrix(y_test, nb_preds)
disp_nb = ConfusionMatrixDisplay(cm_nb, display_labels=["Fake", "Real"])
disp_nb.plot(cmap="Oranges")
plt.title("Naive Bayes — Confusion Matrix")
plt.tight_layout()
plt.savefig("cm_naive_bayes.png")
plt.show()

# ── 7. MODEL COMPARISON BAR CHART ────────────────────────────────
models = ["Logistic Regression", "Naive Bayes"]
accuracies = [lr_acc, nb_acc]
f1_scores = [lr_f1, nb_f1]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4CAF50")
bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score", color="#2196F3")
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Accuracy & F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.bar_label(bars1, fmt="%.4f", padding=3)
ax.bar_label(bars2, fmt="%.4f", padding=3)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ── 8. TOP TF-IDF WORDS PER MODEL ────────────────────────────────
feature_names = np.array(tfidf.get_feature_names_out())

# Top 20 words most associated with Fake vs Real (Logistic Regression coefficients) going to be chart too
top_fake_idx = np.argsort(lr.coef_[0])[:20]  # most negative = predicts most ish  Fake
top_real_idx = np.argsort(lr.coef_[0])[-20:][
    ::-1
]  # most positive = predicts probaly Real

print("\nTop words predicting FAKE:", feature_names[top_fake_idx].tolist())
print("Top words predicting REAL:", feature_names[top_real_idx].tolist())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(feature_names[top_fake_idx], lr.coef_[0][top_fake_idx], color="salmon")
axes[0].set_title("Top 20 Words → Fake News")
axes[0].invert_yaxis()
axes[1].barh(feature_names[top_real_idx], lr.coef_[0][top_real_idx], color="lightgreen")
axes[1].set_title("Top 20 Words → Real News")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("top_words.png")
plt.show()

print("\n Files have finished processing")  # file should have image file in file folder
