"""
YouTube Comment Analyzer
------------------------
Collects the first 200 top-level comments of a YouTube video through the
YouTube Data API v3 and produces an audience-response analysis:

    1. Data collection      - video metadata + 200 comments (relevance order)
    2. Pre-processing       - cleaning, tokenisation, stop-word removal, lemmatisation
    3. Sentiment analysis   - VADER compound score per comment
    4. Keyword analysis     - word frequencies, bigrams, trigrams, TF-IDF
    5. Topic modelling      - LDA, number of topics chosen by UMass coherence
    6. Comment categories   - rule-based detection of questions, requests, praise, etc.
    7. Timeline analysis    - which parts of the video viewers reacted to (quoted timestamps)
    8. Insight generation   - future video ideas and recommendations

Outputs (written next to this file):
    analysis_report.csv     - the full analysis report
    comments_dataset.csv    - every collected comment with its computed features
    charts/*.png            - figures used in the written report

Usage:
    python youtube_comment_analyzer.py
    python youtube_comment_analyzer.py "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import csv
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations

import emoji
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_URL = "https://www.youtube.com/watch?v=nQdXp6HKKik"
MAX_COMMENTS = 200
COMMENT_ORDER = "relevance"        # "relevance" = top comments, "time" = newest
RANDOM_STATE = 42                  # fixed seed so every run gives the same result

# Standard VADER thresholds (Hutto & Gilbert, 2014)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

SEGMENT_MINUTES = 2               # width of the video-timeline buckets
CREATOR_NAMES = {"arun", "maini", "mrwhosetheboss"}   # names are not topics

TOPIC_RANGE = range(3, 8)          # candidate numbers of LDA topics
TOP_WORDS_PER_TOPIC = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")
REPORT_CSV = os.path.join(BASE_DIR, "analysis_report.csv")
DATASET_CSV = os.path.join(BASE_DIR, "comments_dataset.csv")

API_URL = "https://www.googleapis.com/youtube/v3"

DetectorFactory.seed = RANDOM_STATE

# Chart colours: blue = positive, grey = neutral, red = negative
COLOR_POS = "#2a78d6"
COLOR_NEU = "#8a8984"
COLOR_NEG = "#e34948"
COLOR_MAIN = "#2a78d6"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# 1. Data collection
# ---------------------------------------------------------------------------

def extract_video_id(url):
    """Return the 11-character video ID from any common YouTube URL format."""
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/|embed/|live/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not find a video ID in: {url}")


def api_get(endpoint, params, api_key):
    params = dict(params, key=api_key)
    response = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=30)
    if response.status_code != 200:
        try:
            error = response.json()["error"]
            reason = error["errors"][0].get("reason", "")
            message = error.get("message", "")
        except (ValueError, KeyError, IndexError):
            reason, message = "", response.text
        if reason == "commentsDisabled":
            raise RuntimeError("Comments are disabled for this video.")
        if reason == "quotaExceeded":
            raise RuntimeError("Daily API quota exceeded. Try again tomorrow.")
        raise RuntimeError(f"YouTube API error {response.status_code}: {message}")
    return response.json()


def fetch_video_details(video_id, api_key):
    data = api_get("videos", {"part": "snippet,statistics,contentDetails", "id": video_id}, api_key)
    if not data.get("items"):
        raise RuntimeError("Video not found. Check the link.")
    item = data["items"][0]
    snippet, stats = item["snippet"], item.get("statistics", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "duration": item.get("contentDetails", {}).get("duration", ""),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "total_comments": int(stats.get("commentCount", 0)),
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),
    }


def fetch_comments(video_id, api_key, limit=MAX_COMMENTS):
    """Page through commentThreads (100 per request) until `limit` comments are collected."""
    comments, page_token, requests_made = [], None, 0
    while len(comments) < limit:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, limit - len(comments)),
            "order": COMMENT_ORDER,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        data = api_get("commentThreads", params, api_key)
        requests_made += 1

        for thread in data.get("items", []):
            top = thread["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": thread["id"],
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", ""),
                "like_count": int(top.get("likeCount", 0)),
                "reply_count": int(thread["snippet"].get("totalReplyCount", 0)),
                "published_at": top.get("publishedAt", ""),
            })

        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return comments[:limit], requests_made


# ---------------------------------------------------------------------------
# 2. Pre-processing
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TIMESTAMP_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")
MENTION_RE = re.compile(r"@[\w.-]+")
NON_LETTER_RE = re.compile(r"[^a-z\s']")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # "sooooo" -> "soo"

# Informal spellings that are common in YouTube comments
SLANG = {
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please",
    "thx": "thanks", "ty": "thanks", "bc": "because", "cuz": "because",
    "coz": "because", "tho": "though", "w": "with", "vid": "video",
    "vids": "videos", "ppl": "people", "im": "i'm", "dont": "don't",
    "cant": "can't", "wont": "won't", "didnt": "didn't", "doesnt": "doesn't",
    "isnt": "isn't", "thats": "that's", "ive": "i've", "idk": "i don't know",
}

# Extra stop words: filler words that carry no topical meaning in comments
CUSTOM_STOPWORDS = {
    "video", "videos", "like", "get", "got", "one", "would", "also", "really",
    "much", "even", "still", "thing", "things", "know", "think", "see", "make",
    "go", "going", "say", "said", "lot", "way", "well", "yeah", "yes", "oh",
    "lol", "lmao", "omg", "guy", "guys", "bro", "dude", "man", "u", "im",
    "i'm", "it's", "that's", "don't", "can't", "dont", "cant", "gonna",
    "wanna", "let", "us", "could", "every", "ever", "though", "actually",
    "always", "something", "someone", "anyone", "everyone", "thank", "thanks",
    "people", "want", "need", "back", "take", "come", "made", "look", "use",
    "good", "great", "time", "day", "year", "years", "watch", "watching",
    "watched", "comment", "comments", "first", "two", "next",
}

STOPWORDS = set(stopwords.words("english")) | CUSTOM_STOPWORDS | CREATOR_NAMES
LEMMATIZER = WordNetLemmatizer()


# On YouTube these emoji almost always signal laughter, not sadness or death.
# Left alone, VADER reads "loudly crying face" or "skull" as strongly negative.
LAUGH_EMOJI_RE = re.compile("[\U0001F602\U0001F923\U0001F62D\U0001F480\u2620\U0001F606\U0001F639\U0001F605]")

# Internet slang whose meaning differs from the dictionary sense VADER assumes
SLANG_LEXICON = {
    "insane": 1.5, "crazy": 0.5, "goat": 2.0, "dope": 2.0, "lit": 1.5,
    "underrated": 1.5, "fire": 1.0, "sick": 1.5, "haha": 1.5, "ngl": 0.0,
}


def normalise_text(text):
    """Light cleaning used for sentiment and rule matching (keeps punctuation and meaning)."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = LAUGH_EMOJI_RE.sub(" haha ", text)
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
    text = REPEAT_RE.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Heavy cleaning used for keyword and topic analysis."""
    text = URL_RE.sub(" ", text.lower())
    text = MENTION_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_LETTER_RE.sub(" ", text)
    tokens = []
    for raw in text.split():
        word = SLANG.get(raw.strip("'"), raw.strip("'"))
        for part in word.split():
            if len(part) < 3 or part in STOPWORDS:
                continue
            lemma = LEMMATIZER.lemmatize(part)
            lemma = LEMMATIZER.lemmatize(lemma, pos="v")
            if len(lemma) >= 3 and lemma not in STOPWORDS:
                tokens.append(lemma)
    return tokens


def detect_language(text):
    letters = re.sub(r"[^\w\s]", "", emoji.replace_emoji(text, replace="")).strip()
    if len(letters) < 3:
        return "emoji/symbols"
    words = [w for w in TIMESTAMP_RE.sub(" ", letters).split() if w.isalpha()]
    if len(words) < 4:
        return "undetermined (too short)"   # langdetect is unreliable on very short text
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# 3. Sentiment analysis
# ---------------------------------------------------------------------------

def label_sentiment(compound):
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def add_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(SLANG_LEXICON)
    scores = df["normalised_text"].apply(analyzer.polarity_scores)
    df["sent_positive"] = scores.apply(lambda s: s["pos"])
    df["sent_neutral"] = scores.apply(lambda s: s["neu"])
    df["sent_negative"] = scores.apply(lambda s: s["neg"])
    df["compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment"] = df["compound"].apply(label_sentiment)
    return df


# ---------------------------------------------------------------------------
# 4. Keyword analysis
# ---------------------------------------------------------------------------

def ngram_counts(token_lists, n):
    counts = Counter()
    for tokens in token_lists:
        counts.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return counts


def tfidf_terms(docs, top_n=25):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = mean_scores.argsort()[::-1][:top_n]
    return [(terms[i], float(mean_scores[i])) for i in order]


def distinctive_words(pos_tokens, neg_tokens, top_n=10, min_count=2):
    """Words used far more in one group than the other (smoothed log-ratio of relative frequency)."""
    pos_counts, neg_counts = Counter(pos_tokens), Counter(neg_tokens)
    pos_total, neg_total = max(sum(pos_counts.values()), 1), max(sum(neg_counts.values()), 1)
    vocab = sorted(set(pos_counts) | set(neg_counts))      # sorted so ties rank the same way on every run
    v = len(vocab) or 1

    def score(word, a, a_total, b, b_total):
        return math.log((a[word] + 1) / (a_total + v)) - math.log((b[word] + 1) / (b_total + v))

    pos_side = sorted(
        (w for w in vocab if pos_counts[w] >= min_count),
        key=lambda w: score(w, pos_counts, pos_total, neg_counts, neg_total), reverse=True,
    )[:top_n]
    neg_side = sorted(
        (w for w in vocab if neg_counts[w] >= min_count),
        key=lambda w: score(w, neg_counts, neg_total, pos_counts, pos_total), reverse=True,
    )[:top_n]
    return pos_side, neg_side


# ---------------------------------------------------------------------------
# 5. Topic modelling
# ---------------------------------------------------------------------------

def umass_coherence(topic_word_ids, doc_term):
    """UMass coherence (Mimno et al., 2011): higher (closer to 0) is better."""
    binary = (doc_term > 0).astype(int)
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()
    co_doc = (binary.T @ binary).toarray()
    scores = []
    for word_ids in topic_word_ids:
        total = 0.0
        for i, j in combinations(range(len(word_ids)), 2):
            wi, wj = word_ids[i], word_ids[j]       # wi ranks above wj
            total += math.log((co_doc[wi, wj] + 1) / max(doc_freq[wi], 1))
        scores.append(total)
    return float(np.mean(scores))


def run_topic_model(docs):
    vectorizer = CountVectorizer(min_df=2, max_df=0.5)
    doc_term = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    trials = []
    for k in TOPIC_RANGE:
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="batch", max_iter=100,
            random_state=RANDOM_STATE, doc_topic_prior=0.1, topic_word_prior=0.01,
        )
        lda.fit(doc_term)
        top_ids = [comp.argsort()[::-1][:TOP_WORDS_PER_TOPIC] for comp in lda.components_]
        coherence = umass_coherence(top_ids, doc_term)
        trials.append({"k": k, "coherence": coherence, "model": lda, "top_ids": top_ids})

    best = max(trials, key=lambda t: t["coherence"])
    lda = best["model"]
    doc_topic = lda.transform(doc_term)
    topics = [[vocab[i] for i in ids] for ids in best["top_ids"]]
    coherence_table = [(t["k"], t["coherence"]) for t in trials]
    return topics, doc_topic, coherence_table, doc_term.shape


# ---------------------------------------------------------------------------
# 6. Rule-based comment categories
# ---------------------------------------------------------------------------

REQUEST_PATTERNS = [
    r"(?:make|do|need|want|see|have|upload|post)\w*\s+(?:a\s+|another\s+|more\s+)?(?:video|videos|episode|series|content|review)\s+(?:on|about|covering|where)\s+([^.!?\n]{3,70})",
    r"(?:please|pls|plz|kindly)\s+(?:make|do|cover|talk about|react to|explain|upload|try|review)\s+([^.!?\n]{3,70})",
    r"(?:can|could|would|will)\s+you\s+(?:please\s+|pls\s+)?(?:make|do|cover|talk about|react to|explain|try|review|show)\s+([^.!?\n]{3,70})",
    r"(?:you\s+should|u\s+should|should\s+make|should\s+do)\s+([^.!?\n]{3,70})",
    r"(?:would\s+love\s+to\s+see|i'?d\s+love\s+to\s+see|hope\s+you\s+(?:make|do|cover))\s+([^.!?\n]{3,70})",
    r"\b(part\s*(?:2|two|ii)|next\s+part|sequel|follow[\s-]?up)\b",
]
REQUEST_RE = [re.compile(p, re.IGNORECASE) for p in REQUEST_PATTERNS]

PURCHASE_RE = re.compile(
    r"\b(i want (?:one|this|that|it)|i need (?:one|this|that|it)|i'?d (?:buy|get)|i (?:will|would) buy|take my money|"
    r"where (?:can|do) i (?:buy|get)|how much (?:is|does)|ordered (?:one|it)|just bought|want to buy|wanna buy|"
    r"want one|need one)\b",
    re.IGNORECASE,
)

QUESTION_START_RE = re.compile(
    r"^(?:who|what|when|where|why|how|which|is|are|can|could|does|do|did|will|would|should)\b",
    re.IGNORECASE,
)
APPRECIATION_RE = re.compile(
    r"\b(thank|thanks|thx|grateful|appreciate|love (?:this|your|it|you)|best (?:video|channel|explanation|youtuber)|"
    r"amazing|awesome|incredible|brilliant|masterpiece|underrated|legend|goat|beautiful|helpful|well done|great job)\b",
    re.IGNORECASE,
)
CRITICISM_RE = re.compile(
    r"\b(clickbait|misleading|boring|too long|too short|annoying|wrong|incorrect|inaccurate|disappoint\w*|"
    r"waste|cringe|worst|hate|bad audio|audio|volume|sound quality|music too loud|too many ads|sponsor|"
    r"unsubscrib\w*|dislike|mistake|error|biased|fake)\b",
    re.IGNORECASE,
)
HUMOR_RE = re.compile(
    r"\b(haha|lol|lmao|lmfao|rofl|haha\w*|hehe\w*|funny|hilarious|dying|i'm dead|im dead)\b|"
    r"face with tears of joy|rolling on the floor laughing|skull",
    re.IGNORECASE,
)
PERSONAL_RE = re.compile(
    r"\b(i remember|when i was|my (?:mom|dad|son|daughter|kid|family|friend|wife|husband|brother|sister|teacher)|"
    r"i used to|as a kid|growing up|in my experience|i grew up|i'm from|i am from|my country)\b",
    re.IGNORECASE,
)


def extract_requests(text):
    found = []
    for pattern in REQUEST_RE:
        for match in pattern.finditer(text):
            snippet = match.group(1).strip(" ,;:-\"'")
            if snippet:
                found.append(snippet)
    return found


def is_question(text):
    stripped = text.strip()
    return "?" in stripped or bool(QUESTION_START_RE.match(stripped))


def categorise(df):
    df["timestamps"] = df["text"].apply(lambda t: ", ".join(TIMESTAMP_RE.findall(t)))
    df["requests"] = df["text"].apply(lambda t: " | ".join(extract_requests(t)))
    df["is_request"] = df["requests"] != ""
    df["is_question"] = df["text"].apply(is_question)
    df["is_purchase_intent"] = df["text"].apply(lambda t: bool(PURCHASE_RE.search(t)))
    df["is_appreciation"] = df["normalised_text"].apply(lambda t: bool(APPRECIATION_RE.search(t)))
    df["is_humor"] = df["normalised_text"].apply(lambda t: bool(HUMOR_RE.search(t)))
    # Criticism = explicit critical vocabulary, or a negative comment that is not a joke
    df["is_criticism"] = df.apply(
        lambda r: (bool(CRITICISM_RE.search(r["text"])) and r["compound"] < 0.3)
        or (r["sentiment"] == "Negative" and not r["is_humor"]),
        axis=1,
    )
    df["is_personal_story"] = df["text"].apply(lambda t: bool(PERSONAL_RE.search(t)))
    df["mentions_timestamp"] = df["timestamps"] != ""

    def primary(r):
        if r["is_request"]:
            return "Request / Suggestion"
        if r["is_purchase_intent"]:
            return "Purchase Intent"
        if r["is_criticism"]:
            return "Criticism / Concern"
        if r["is_question"]:
            return "Question"
        if r["is_personal_story"]:
            return "Personal Story"
        if r["is_appreciation"]:
            return "Appreciation / Praise"
        if r["is_humor"]:
            return "Humour / Joke"
        return "General Opinion"

    df["category"] = df.apply(primary, axis=1)
    return df


# ---------------------------------------------------------------------------
# 6b. Video timeline analysis
# ---------------------------------------------------------------------------

def parse_iso_duration(value):
    match = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not match:
        return 0
    d, h, m, s = (int(x) if x else 0 for x in match.groups())
    return d * 86400 + h * 3600 + m * 60 + s


def timestamp_to_seconds(stamp):
    parts = [int(p) for p in stamp.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def timeline_segments(df, duration_seconds):
    """Group comments by the first timestamp they mention into fixed-width video segments."""
    width = SEGMENT_MINUTES * 60
    limit = duration_seconds or 10 ** 6
    rows = []
    for _, r in df[df["mentions_timestamp"]].iterrows():
        seconds = timestamp_to_seconds(r["timestamps"].split(", ")[0])
        if seconds <= limit:
            rows.append((seconds // width, r))
    segments = []
    for bucket in sorted({b for b, _ in rows}):
        members = pd.DataFrame([r for b, r in rows if b == bucket])
        tokens = [t for toks in members["tokens"] for t in toks]
        start = bucket * width
        segments.append({
            "label": f"{start // 60:02d}:{start % 60:02d}-{(start + width) // 60:02d}:{(start + width) % 60:02d}",
            "start": start,
            "count": len(members),
            "likes": int(members["like_count"].sum()),
            "avg_compound": float(members["compound"].mean()),
            "keywords": [w for w, _ in Counter(tokens).most_common(5)],
            "example": members.sort_values("like_count", ascending=False)["text"].iloc[0],
        })
    return segments


# ---------------------------------------------------------------------------
# 7. Insight generation
# ---------------------------------------------------------------------------

def response_verdict(pos_share, neg_share, mean_compound):
    if pos_share >= 0.6 and neg_share <= 0.15:
        return "Very positive - the audience strongly approves of this video."
    if pos_share >= 0.45 and pos_share > 2 * neg_share:
        return "Positive - most viewers responded well, with limited criticism."
    if neg_share >= 0.35 and neg_share > pos_share:
        return "Negative - a large share of viewers reacted critically."
    if abs(pos_share - neg_share) < 0.1:
        return "Polarised - the audience is split between approval and criticism."
    return "Mixed to neutral - reactions are moderate, with no strong lean."


def topic_name(words):
    return " / ".join(words[:3])


def build_future_ideas(topic_stats, request_counter, question_rows, segments, purchase_rows):
    ideas = []
    for snippet, count in request_counter.most_common(6):
        ideas.append((
            f"Direct viewer request: \"{snippet}\"",
            f"Asked for in {count} comment(s). Explicit requests are the most reliable signal of demand.",
        ))
    # Themes ranked by engagement adjusted for tone
    ranked = sorted(topic_stats, key=lambda t: (t["avg_likes"] * (1 + t["avg_compound"])), reverse=True)
    for t in ranked[:3]:
        ideas.append((
            f"Deeper follow-up on the theme \"{t['name']}\"",
            f"{t['share']:.0%} of comments discuss this theme (average sentiment {t['avg_compound']:+.2f}, "
            f"average {t['avg_likes']:.1f} likes per comment). Key terms: {', '.join(t['words'][:6])}. "
            f"Most-liked example: \"{shorten(t['example'], 120)}\"",
        ))
    # The parts of the video that pulled the most liked reactions
    for seg in sorted(segments, key=lambda s: s["likes"], reverse=True)[:2]:
        ideas.append((
            f"Expand the segment at {seg['label']} into its own video",
            f"{seg['count']} timestamped comments with {seg['likes']} likes in total point here "
            f"(keywords: {', '.join(seg['keywords'][:4])}). Example: \"{shorten(seg['example'], 120)}\"",
        ))
    if len(purchase_rows) >= 3:
        ideas.append((
            "Long-term review of the products viewers want to buy",
            f"{len(purchase_rows)} comments show purchase intent, e.g. \"{shorten(purchase_rows.iloc[0]['text'], 120)}\". "
            "A follow-up after real-world use answers the buying question directly.",
        ))
    # Genuine questions only: skip jokes and one-liners
    genuine = question_rows[
        (~question_rows["is_humor"])
        & (question_rows["word_count"] >= 6)
        & question_rows["text"].str.contains(
            r"\b(?:how|why|what|which|can you|will you|could you|is it|does it)\b", case=False, regex=True)
    ]
    for _, row in genuine.head(2).iterrows():
        ideas.append((
            "Answer an open audience question in a dedicated video or Q&A segment",
            f"\"{shorten(row['text'], 140)}\" ({row['like_count']} likes).",
        ))
    return ideas


def build_recommendations(df, pos_words, neg_words, topic_stats, corr, segments):
    recs = []
    pos_share = (df["sentiment"] == "Positive").mean()
    neg = df[df["sentiment"] == "Negative"]

    if pos_words:
        recs.append((
            "Keep the elements viewers praise",
            f"Words that appear far more often in positive comments: {', '.join(pos_words[:8])}. "
            "These describe what the audience values and should remain a consistent part of future videos.",
        ))
    if len(neg) and neg_words:
        recs.append((
            "Address the recurring concerns",
            f"Words that appear far more often in negative comments: {', '.join(neg_words[:8])}. "
            f"{len(neg)} comments ({len(neg) / len(df):.0%}) were negative; reviewing them highlights what to avoid.",
        ))
    crit = df[df["is_criticism"]].sort_values("like_count", ascending=False)
    if len(crit):
        recs.append((
            "Respond to the most-liked criticism",
            f"\"{shorten(crit.iloc[0]['text'], 140)}\" ({crit.iloc[0]['like_count']} likes). "
            "Acknowledging well-supported criticism publicly (pinned reply or follow-up) protects trust.",
        ))
    if df["is_question"].sum():
        recs.append((
            "Reply to viewer questions in the comments",
            f"{int(df['is_question'].sum())} comments contain questions. Answering them (or pinning a FAQ reply) "
            "raises engagement and shows the creator listens.",
        ))
    if segments:
        hot = sorted(segments, key=lambda s: s["likes"], reverse=True)[:3]
        recs.append((
            "Reuse the format of the moments viewers reference",
            f"{int(df['mentions_timestamp'].sum())} comments ({df['mentions_timestamp'].mean():.0%}) quote a timestamp, "
            "so viewers watch closely and react to specific moments. The strongest segments were "
            + "; ".join(f"{s['label']} ({s['likes']} likes, {', '.join(s['keywords'][:3])})" for s in hot)
            + ". These are good candidates for Shorts, thumbnails and similar segments in future videos.",
        ))
    if df["is_humor"].mean() >= 0.08:
        recs.append((
            "Keep the humour",
            f"{df['is_humor'].mean():.0%} of comments react with jokes or laughter; the light tone is part of the appeal.",
        ))
    if df["is_personal_story"].sum() >= 3:
        recs.append((
            "Invite viewers to share their own experiences",
            f"{int(df['is_personal_story'].sum())} viewers shared personal stories. A call-to-action asking for "
            "experiences turns passive viewers into active commenters.",
        ))
    best = max(topic_stats, key=lambda t: t["avg_compound"])
    recs.append((
        "Lean into the best-received theme",
        f"The theme \"{best['name']}\" has the highest average sentiment ({best['avg_compound']:+.2f}).",
    ))
    if not math.isnan(corr):
        direction = "more" if corr > 0 else "less"
        strength = "weak" if abs(corr) < 0.3 else ("moderate" if abs(corr) < 0.6 else "strong")
        recs.append((
            "Understand what the community upvotes",
            f"Spearman correlation between sentiment and likes is {corr:+.2f} ({strength}); "
            f"positive comments tend to receive {direction} likes than negative ones."
            if abs(corr) >= 0.1 else
            f"Spearman correlation between sentiment and likes is {corr:+.2f}; likes are driven by content "
            "(insight, humour, relatability) rather than tone alone.",
        ))
    if pos_share < 0.5:
        recs.append((
            "Rebuild positive momentum",
            "Fewer than half the comments are positive. Clarify the video's purpose early, fix the issues raised, "
            "and engage directly with critical viewers.",
        ))
    return recs


def shorten(text, limit):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def style_axes(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color=COLOR_INK, loc="left", pad=12)
    ax.set_xlabel(xlabel, fontsize=9, color=COLOR_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hbar(items, title, xlabel, name, color=COLOR_MAIN, fmt="{:.0f}"):
    labels = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(items) + 1)))
    bars = ax.barh(labels, values, color=color, height=0.65)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt.format(value),
                va="center", fontsize=8, color=COLOR_MUTED)
    style_axes(ax, title, xlabel)
    save(fig, name)


def make_charts(df, unigrams, bigrams, tfidf, topic_stats, coherence_table, best_k, segments):
    os.makedirs(CHART_DIR, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Sentiment distribution
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(order, counts.values, color=[COLOR_POS, COLOR_NEU, COLOR_NEG], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value}  ({value / len(df):.0%})",
                ha="center", va="bottom", fontsize=9, color=COLOR_INK)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    style_axes(ax, "Sentiment distribution of comments", ylabel="Number of comments")
    save(fig, "01_sentiment_distribution.png")

    # Compound score histogram
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(-1, 1, 21)
    for label, color in zip(order, [COLOR_POS, COLOR_NEU, COLOR_NEG]):
        ax.hist(df.loc[df["sentiment"] == label, "compound"], bins=bins, color=color,
                edgecolor="white", linewidth=1, label=label)
    ax.axvline(df["compound"].mean(), color=COLOR_INK, linewidth=1.2, linestyle="--")
    ax.text(df["compound"].mean(), ax.get_ylim()[1] * 0.95, f"  mean = {df['compound'].mean():+.2f}",
            fontsize=8, color=COLOR_INK, va="top")
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "Distribution of VADER compound scores", "Compound score (-1 = very negative, +1 = very positive)",
               "Number of comments")
    save(fig, "02_compound_histogram.png")

    hbar(unigrams[:20], "Top 20 most frequent words", "Frequency", "03_top_words.png")
    hbar(bigrams[:15], "Top 15 most frequent two-word phrases", "Frequency", "04_top_bigrams.png")
    hbar(tfidf[:15], "Top 15 key terms by TF-IDF weight", "Mean TF-IDF score", "05_tfidf_terms.png", fmt="{:.3f}")

    # Topic coherence
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ks, scores = zip(*coherence_table)
    ax.plot(ks, scores, color=COLOR_MAIN, linewidth=2, marker="o", markersize=8)
    ax.scatter([best_k], [dict(coherence_table)[best_k]], s=140, facecolor="none", edgecolor=COLOR_INK, linewidth=1.5, zorder=3)
    ax.annotate(f"chosen: {best_k} topics", (best_k, dict(coherence_table)[best_k]),
                textcoords="offset points", xytext=(10, -14), fontsize=8, color=COLOR_INK)
    ax.set_xticks(list(ks))
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    style_axes(ax, "Topic model selection (UMass coherence)", "Number of topics (k)", "Coherence (higher is better)")
    save(fig, "06_topic_coherence.png")

    # Topic share with sentiment colour
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(topic_stats) + 1.2))
    labels = [f"T{t['id']}: {t['name']}" for t in topic_stats][::-1]
    shares = [t["count"] for t in topic_stats][::-1]
    colors = [COLOR_POS if t["avg_compound"] >= POSITIVE_THRESHOLD else
              (COLOR_NEG if t["avg_compound"] <= NEGATIVE_THRESHOLD else COLOR_NEU) for t in topic_stats][::-1]
    bars = ax.barh(labels, shares, color=colors, height=0.6)
    for bar, t in zip(bars, topic_stats[::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {t['count']} comments, sentiment {t['avg_compound']:+.2f}", va="center", fontsize=8, color=COLOR_MUTED)
    ax.set_xlim(0, max(shares) * 1.6)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    style_axes(ax, "Discussion topics (colour = average sentiment)", "Number of comments")
    save(fig, "07_topics.png")

    # Comment categories
    cats = df["category"].value_counts()
    hbar(list(cats.items()), "Comment categories", "Number of comments", "08_categories.png")

    # Likes vs sentiment
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for label, color in zip(order, [COLOR_POS, COLOR_NEU, COLOR_NEG]):
        sub = df[df["sentiment"] == label]
        ax.scatter(sub["compound"], sub["like_count"] + 1, s=36, color=color, alpha=0.8,
                   edgecolor="white", linewidth=1, label=label)
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "Comment likes versus sentiment", "Compound score", "Likes + 1 (log scale)")
    save(fig, "09_likes_vs_sentiment.png")

    # Comment length
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cap = int(np.percentile(df["word_count"], 98)) + 5     # keep a few very long comments from squashing the chart
    ax.hist(df["word_count"].clip(upper=cap), bins=range(0, cap + 5, 5), color=COLOR_MAIN, edgecolor="white", linewidth=1)
    ax.text(0.98, 0.92, f"median = {df['word_count'].median():.0f} words\nlongest = {df['word_count'].max()} words",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=COLOR_MUTED)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    style_axes(ax, "Comment length distribution", "Words per comment", "Number of comments")
    save(fig, "10_comment_length.png")

    # Video timeline: which segments viewers reacted to
    if segments:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        labels = [s["label"].split("-")[0] for s in segments]
        colors = [COLOR_POS if s["avg_compound"] >= POSITIVE_THRESHOLD else
                  (COLOR_NEG if s["avg_compound"] <= NEGATIVE_THRESHOLD else COLOR_NEU) for s in segments]
        bars = ax.bar(labels, [s["count"] for s in segments], color=colors, width=0.7)
        for bar, s in zip(bars, segments):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(s["count"]),
                    ha="center", va="bottom", fontsize=8, color=COLOR_MUTED)
        ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COLOR_POS, COLOR_NEU, COLOR_NEG)]
        ax.legend(handles, ["Positive avg", "Neutral avg", "Negative avg"], frameon=False, fontsize=8,
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
        style_axes(ax, f"Timestamped comments per {SEGMENT_MINUTES}-minute video segment",
                   "Segment start (mm:ss)", "Number of comments")
        save(fig, "11_video_timeline.png")


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(rows):
    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Item", "Value", "Details"])
        writer.writerows(rows)


def main():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        sys.exit("Add your YouTube Data API key to the .env file first (YOUTUBE_API_KEY=...).")

    url = sys.argv[1] if len(sys.argv) > 1 else VIDEO_URL
    video_id = extract_video_id(url)

    print(f"[1/7] Fetching video details for {video_id} ...")
    video = fetch_video_details(video_id, api_key)
    print(f"      {video['title']}  |  {video['channel']}")

    print(f"[2/7] Fetching up to {MAX_COMMENTS} comments ({COMMENT_ORDER} order) ...")
    raw_comments, api_calls = fetch_comments(video_id, api_key)
    if not raw_comments:
        sys.exit("No comments were returned for this video.")
    df = pd.DataFrame(raw_comments)
    df.insert(0, "rank", range(1, len(df) + 1))
    print(f"      {len(df)} comments collected with {api_calls + 1} API requests")

    print("[3/7] Pre-processing text ...")
    df["normalised_text"] = df["text"].apply(normalise_text)
    df["tokens"] = df["text"].apply(tokenize)
    df["clean_text"] = df["tokens"].apply(" ".join)
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))
    df["token_count"] = df["tokens"].apply(len)
    df["emoji_count"] = df["text"].apply(emoji.emoji_count)
    df["language"] = df["text"].apply(detect_language)
    raw_word_total = int(df["word_count"].sum())
    token_total = int(df["token_count"].sum())

    print("[4/7] Scoring sentiment (VADER) ...")
    df = add_sentiment(df)

    print("[5/7] Extracting keywords and phrases ...")
    all_tokens = [tok for tokens in df["tokens"] for tok in tokens]
    unigrams = Counter(all_tokens).most_common(30)
    bigrams = [(k, v) for k, v in ngram_counts(df["tokens"], 2).most_common(20) if v >= 2]
    trigrams = [(k, v) for k, v in ngram_counts(df["tokens"], 3).most_common(15) if v >= 2]
    docs = df["clean_text"].where(df["clean_text"] != "", "")
    usable = docs[docs.str.len() > 0]
    tfidf = tfidf_terms(usable.tolist())
    pos_tok = [t for toks in df.loc[df["sentiment"] == "Positive", "tokens"] for t in toks]
    neg_tok = [t for toks in df.loc[df["sentiment"] == "Negative", "tokens"] for t in toks]
    pos_words, neg_words = distinctive_words(pos_tok, neg_tok)

    print("[6/7] Building topic model (LDA) ...")
    topics, doc_topic, coherence_table, dtm_shape = run_topic_model(usable.tolist())
    best_k = len(topics)
    df["topic"] = 0
    df["topic_confidence"] = 0.0
    df.loc[usable.index, "topic"] = doc_topic.argmax(axis=1) + 1
    df.loc[usable.index, "topic_confidence"] = doc_topic.max(axis=1).round(3)
    topic_stats = []
    for i, words in enumerate(topics, start=1):
        sub = df[df["topic"] == i]
        topic_stats.append({
            "id": i, "words": words, "name": topic_name(words), "count": len(sub),
            "share": len(sub) / len(df),
            "avg_compound": float(sub["compound"].mean()) if len(sub) else 0.0,
            "avg_likes": float(sub["like_count"].mean()) if len(sub) else 0.0,
            "pos_share": float((sub["sentiment"] == "Positive").mean()) if len(sub) else 0.0,
            "example": sub.sort_values("like_count", ascending=False)["text"].iloc[0] if len(sub) else "",
        })

    print("[7/7] Categorising comments and generating insights ...")
    df = categorise(df)
    request_counter = Counter(
        r.strip().lower() for reqs in df["requests"] for r in reqs.split(" | ") if r.strip()
    )
    questions = df[df["is_question"]].sort_values("like_count", ascending=False)
    purchase = df[df["is_purchase_intent"]].sort_values("like_count", ascending=False)
    segments = timeline_segments(df, parse_iso_duration(video["duration"]))
    corr = df["compound"].corr(df["like_count"], method="spearman")

    # ----- assemble report rows -----
    n = len(df)
    sent_counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    pos_share, neg_share = sent_counts["Positive"] / n, sent_counts["Negative"] / n
    weights = np.log1p(df["like_count"]) + 1
    weighted_compound = float(np.average(df["compound"], weights=weights))
    published = datetime.fromisoformat(video["published_at"].replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - published).days

    R = []
    sec = "1. Video Overview"
    R += [
        [sec, "Title", video["title"], ""],
        [sec, "Channel", video["channel"], ""],
        [sec, "Video ID", video_id, f"https://www.youtube.com/watch?v={video_id}"],
        [sec, "Published", published.strftime("%Y-%m-%d"), f"{age_days} days before analysis"],
        [sec, "Duration (ISO 8601)", video["duration"], ""],
        [sec, "Views", video["views"], ""],
        [sec, "Likes", video["likes"], f"Like-to-view ratio {video['likes'] / max(video['views'], 1):.2%}"],
        [sec, "Total comments on video", video["total_comments"],
         f"Comment-to-view ratio {video['total_comments'] / max(video['views'], 1):.3%}"],
    ]
    sec = "2. Data Collection"
    R += [
        [sec, "Source", "YouTube Data API v3", "endpoints: videos.list, commentThreads.list"],
        [sec, "Comments analysed", n, f"Top-level comments, '{COMMENT_ORDER}' order (YouTube's top comments)"],
        [sec, "API requests used", api_calls + 1, f"Approx. {api_calls + 1} quota units of 10,000 daily"],
        [sec, "Analysis date", datetime.now().strftime("%Y-%m-%d %H:%M"), ""],
        [sec, "Total words (raw)", raw_word_total, f"Average {raw_word_total / n:.1f} words per comment"],
        [sec, "Tokens after cleaning", token_total,
         f"{1 - token_total / max(raw_word_total, 1):.0%} removed as stop words, noise or short tokens"],
        [sec, "Unique vocabulary", len(set(all_tokens)), "Distinct lemmatised words"],
        [sec, "Comments containing emoji", int((df["emoji_count"] > 0).sum()), f"{int(df['emoji_count'].sum())} emoji in total"],
    ]
    for lang, count in df["language"].value_counts().head(6).items():
        R.append([sec, f"Language: {lang}", count, f"{count / n:.0%} of comments"])

    sec = "3. Overall Audience Response"
    R += [
        [sec, "Verdict", response_verdict(pos_share, neg_share, df["compound"].mean()), ""],
        [sec, "Positive comments", int(sent_counts["Positive"]), f"{pos_share:.1%}"],
        [sec, "Neutral comments", int(sent_counts["Neutral"]), f"{sent_counts['Neutral'] / n:.1%}"],
        [sec, "Negative comments", int(sent_counts["Negative"]), f"{neg_share:.1%}"],
        [sec, "Mean compound score", round(df["compound"].mean(), 3), "Range -1 (very negative) to +1 (very positive)"],
        [sec, "Median compound score", round(df["compound"].median(), 3), ""],
        [sec, "Like-weighted compound score", round(weighted_compound, 3),
         "Each comment weighted by log(1 + likes) + 1, so popular opinions count more"],
        [sec, "Positive-to-negative ratio", round(sent_counts["Positive"] / max(sent_counts["Negative"], 1), 2), ""],
    ]

    sec = "4. Engagement Insights"
    top_liked = df.sort_values("like_count", ascending=False)
    R += [
        [sec, "Total likes on analysed comments", int(df["like_count"].sum()), ""],
        [sec, "Average likes per comment", round(df["like_count"].mean(), 1), f"Median {df['like_count'].median():.0f}"],
        [sec, "Comments with replies", int((df["reply_count"] > 0).sum()), f"{int(df['reply_count'].sum())} replies in total"],
        [sec, "Share of likes held by top 10 comments",
         f"{top_liked['like_count'].head(10).sum() / max(df['like_count'].sum(), 1):.0%}",
         "Shows how concentrated audience agreement is"],
        [sec, "Average sentiment of top 10 liked comments", round(top_liked["compound"].head(10).mean(), 3), ""],
        [sec, "Sentiment vs likes (Spearman)", round(corr, 3) if not math.isnan(corr) else "n/a",
         "Positive value = more positive comments receive more likes"],
        [sec, "Average likes - positive comments", round(df.loc[df["sentiment"] == "Positive", "like_count"].mean(), 1), ""],
        [sec, "Average likes - neutral comments", round(df.loc[df["sentiment"] == "Neutral", "like_count"].mean(), 1), ""],
        [sec, "Average likes - negative comments",
         round(df.loc[df["sentiment"] == "Negative", "like_count"].mean(), 1) if neg_share else 0, ""],
    ]

    sec = "5. Top Keywords"
    for rank, (word, count) in enumerate(unigrams, start=1):
        R.append([sec, f"#{rank}", word, f"{count} occurrences"])
    sec = "6. Top Phrases"
    for phrase, count in bigrams:
        R.append([sec, "Bigram", phrase, f"{count} occurrences"])
    for phrase, count in trigrams:
        R.append([sec, "Trigram", phrase, f"{count} occurrences"])
    sec = "7. TF-IDF Key Terms"
    for rank, (term, score) in enumerate(tfidf, start=1):
        R.append([sec, f"#{rank}", term, f"mean TF-IDF {score:.4f}"])

    sec = "8. Discussion Topics (LDA)"
    for k, score in coherence_table:
        R.append([sec, f"Coherence for k={k}", round(score, 3), "chosen" if k == best_k else ""])
    for t in topic_stats:
        R.append([sec, f"Topic {t['id']}: {t['name']}", f"{t['count']} comments ({t['share']:.0%})",
                  f"Keywords: {', '.join(t['words'])} | avg sentiment {t['avg_compound']:+.2f} | "
                  f"{t['pos_share']:.0%} positive | avg likes {t['avg_likes']:.1f} | "
                  f"example: \"{shorten(t['example'], 150)}\""])

    sec = "9. Comment Categories"
    for cat, count in df["category"].value_counts().items():
        avg = df.loc[df["category"] == cat, "compound"].mean()
        R.append([sec, cat, count, f"{count / n:.0%} of comments | avg sentiment {avg:+.2f}"])
    for flag, label in [("is_question", "Contains a question"), ("is_request", "Contains a request"),
                        ("is_appreciation", "Expresses appreciation"), ("is_criticism", "Raises criticism"),
                        ("is_purchase_intent", "Shows purchase intent"), ("is_humor", "Humour / laughter"), ("is_personal_story", "Shares a personal story"),
                        ("mentions_timestamp", "References a timestamp")]:
        R.append([sec, f"(overlapping) {label}", int(df[flag].sum()), "A comment can carry several of these flags"])

    sec = "10. Most Liked Comments"
    for _, r in top_liked.head(10).iterrows():
        R.append([sec, f"{r['like_count']} likes", r["sentiment"], shorten(r["text"], 300)])
    sec = "11. Most Positive Comments"
    for _, r in df.sort_values("compound", ascending=False).head(5).iterrows():
        R.append([sec, f"compound {r['compound']:+.3f}", f"{r['like_count']} likes", shorten(r["text"], 300)])
    sec = "12. Most Negative Comments"
    for _, r in df.sort_values("compound").head(5).iterrows():
        if r["compound"] <= NEGATIVE_THRESHOLD:
            R.append([sec, f"compound {r['compound']:+.3f}", f"{r['like_count']} likes", shorten(r["text"], 300)])

    sec = "13. What the Audience Loves"
    R.append([sec, "Words distinctive of positive comments", ", ".join(pos_words), "Smoothed log-ratio vs negative comments"])
    for _, r in df[df["is_appreciation"]].sort_values("like_count", ascending=False).head(5).iterrows():
        R.append([sec, "Appreciation example", f"{r['like_count']} likes", shorten(r["text"], 300)])
    sec = "14. Concerns and Criticism"
    R.append([sec, "Words distinctive of negative comments", ", ".join(neg_words) or "none", "Smoothed log-ratio vs positive comments"])
    crit = df[df["is_criticism"]].sort_values("like_count", ascending=False)
    for _, r in crit.head(6).iterrows():
        R.append([sec, "Concern example", f"{r['like_count']} likes | {r['sentiment']}", shorten(r["text"], 300)])

    sec = "15. Audience Requests"
    if request_counter:
        for snippet, count in request_counter.most_common(15):
            R.append([sec, "Requested", snippet, f"{count} mention(s)"])
    else:
        R.append([sec, "Requested", "No explicit requests detected", ""])
    sec = "16. Audience Questions"
    for _, r in questions.head(10).iterrows():
        R.append([sec, f"{r['like_count']} likes", r["sentiment"], shorten(r["text"], 300)])

    sec = "17. Video Timeline Engagement"
    if segments:
        R.append([sec, "Comments quoting a timestamp", int(df["mentions_timestamp"].sum()),
                  f"Grouped into {SEGMENT_MINUTES}-minute segments of the video by the first timestamp quoted"])
        for seg in segments:
            R.append([sec, f"Segment {seg['label']}", f"{seg['count']} comments | {seg['likes']} likes",
                      f"avg sentiment {seg['avg_compound']:+.2f} | keywords: {', '.join(seg['keywords'])} | "
                      f"top comment: \"{shorten(seg['example'], 150)}\""])
    else:
        R.append([sec, "None", "No timestamps referenced", ""])

    sec = "18. Purchase Intent"
    if len(purchase):
        for _, r in purchase.head(10).iterrows():
            R.append([sec, f"{r['like_count']} likes", r["sentiment"], shorten(r["text"], 300)])
    else:
        R.append([sec, "None", "No purchase intent detected", ""])

    sec = "19. Future Video Ideas"
    ideas = build_future_ideas(topic_stats, request_counter, questions, segments, purchase)
    for i, (idea, why) in enumerate(ideas, start=1):
        R.append([sec, f"Idea {i}", idea, why])
    sec = "20. How to Maintain a Good Response"
    recs = build_recommendations(df, pos_words, neg_words, topic_stats, corr, segments)
    for i, (rec, why) in enumerate(recs, start=1):
        R.append([sec, f"Recommendation {i}", rec, why])

    write_report(R)

    dataset_cols = [
        "rank", "comment_id", "author", "published_at", "like_count", "reply_count", "text",
        "clean_text", "word_count", "token_count", "emoji_count", "language",
        "sent_positive", "sent_neutral", "sent_negative", "compound", "sentiment",
        "topic", "topic_confidence", "category", "is_question", "is_request", "is_purchase_intent", "is_appreciation",
        "is_criticism", "is_humor", "is_personal_story", "mentions_timestamp", "timestamps", "requests",
    ]
    df[dataset_cols].to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")

    make_charts(df, unigrams, bigrams, tfidf, topic_stats, coherence_table, best_k, segments)

    print("\nDone.")
    print(f"  Verdict          : {response_verdict(pos_share, neg_share, df['compound'].mean())}")
    print(f"  Sentiment        : {pos_share:.0%} positive, {sent_counts['Neutral'] / n:.0%} neutral, {neg_share:.0%} negative")
    print(f"  Topics           : {best_k} (UMass coherence)")
    print(f"  Report           : {REPORT_CSV}")
    print(f"  Comment dataset  : {DATASET_CSV}")
    print(f"  Charts           : {CHART_DIR}")


if __name__ == "__main__":
    main()
