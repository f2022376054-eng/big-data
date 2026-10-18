# YouTube Comment Analysis

Analyses the first 200 top-level comments of a YouTube video and produces an audience-response report.

The program collects comments through the YouTube Data API v3 and applies:

- text cleaning, stop-word removal and lemmatisation
- VADER sentiment analysis (adapted for YouTube slang and laughter emoji)
- word frequency, n-gram and TF-IDF keyword analysis
- LDA topic modelling, with the number of topics chosen by UMass coherence
- rule-based comment categories (questions, requests, criticism, humour, etc.)
- a timeline analysis of which video segments viewers reacted to

## Files

| File | Description |
|---|---|
| `youtube_comment_analyzer.py` | The analysis program |
| `analysis_report.csv` | Full analysis report (audience response, keywords, topics, future video ideas, recommendations) |
| `comments_dataset.csv` | All 200 comments with their computed features |
| `charts/` | Figures produced by the program |
| `YouTube_Comment_Analysis_Report.docx` | Detailed project report |

## Running

```
pip install requests python-dotenv pandas numpy nltk scikit-learn matplotlib emoji langdetect
python -c "import nltk; [nltk.download(p) for p in ['vader_lexicon','stopwords','wordnet']]"
```

Create a `.env` file in the project folder containing your YouTube Data API key:

```
YOUTUBE_API_KEY=your_key_here
```

Then run:

```
python youtube_comment_analyzer.py
python youtube_comment_analyzer.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Case-study video: *I tested the most Futuristic Tech - it scared me.* (Mrwhosetheboss)
