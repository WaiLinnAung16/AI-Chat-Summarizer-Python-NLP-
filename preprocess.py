# Preprocess text before TF-IDF: lowercase, drop punctuation, remove common English words.
import string

import nltk
from nltk.corpus import stopwords

# Filled on first use so we do not reload stopwords every call.
STOPWORDS = None
_NLTK_DOWNLOADED = False


def download_nltk_if_needed():
    """Download tokenizer and stopword lists once (NLTK needs them on disk)."""
    global _NLTK_DOWNLOADED
    if _NLTK_DOWNLOADED:
        return
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    _NLTK_DOWNLOADED = True


def get_stopwords():
    """Return a set of English stop words (cached after first call)."""
    global STOPWORDS
    if STOPWORDS is None:
        download_nltk_if_needed()
        STOPWORDS = set(stopwords.words("english"))
    return STOPWORDS


def preprocess_text(text):
    """Prepare one sentence for TF-IDF: lowercase, no punctuation, no stop words."""
    text = text.lower()
    remove_punctuation = str.maketrans("", "", string.punctuation)
    text = text.translate(remove_punctuation)
    words = text.split()
    stops = get_stopwords()
    kept = []
    for word in words:
        if word not in stops and len(word) > 0:
            kept.append(word)
    return " ".join(kept)
