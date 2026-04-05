# Pick the strongest sentences using TF-IDF scores (extractive summary).
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import download_nltk_if_needed, preprocess_text


def summarize_tfidf(text, num_sentences=5):
    """
    Split text into sentences, score each sentence with TF-IDF, return the best ones.
    Output sentences are copied from the original text (same spelling and punctuation).
    """
    download_nltk_if_needed()

    sentences = sent_tokenize(text)
    # Drop empty lines after splitting
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) > 0:
            clean_sentences.append(s)

    if len(clean_sentences) == 0:
        return []

    # Build a preprocessed version of each sentence for the vectorizer only
    preprocessed_list = []
    for s in clean_sentences:
        preprocessed_list.append(preprocess_text(s))

    # Keep sentences that still have words after preprocessing
    original_for_scoring = []
    preprocessed_for_scoring = []
    for i in range(len(clean_sentences)):
        if len(preprocessed_list[i].strip()) > 0:
            original_for_scoring.append(clean_sentences[i])
            preprocessed_for_scoring.append(preprocessed_list[i])

    if len(preprocessed_for_scoring) == 0:
        # Everything was stop words; fall back to first N raw sentences
        limit = min(num_sentences, len(clean_sentences))
        return clean_sentences[0:limit]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_for_scoring)

    # One score per sentence: add up TF-IDF weights in that row
    num_rows = tfidf_matrix.shape[0]
    scores = []
    for row_index in range(num_rows):
        row = tfidf_matrix[row_index]
        total = row.sum()
        scores.append(float(total))

    # Sort sentence indexes from highest score to lowest
    ranked_indexes = []
    for i in range(len(scores)):
        ranked_indexes.append(i)

    def sort_key(idx):
        return scores[idx]

    ranked_indexes.sort(key=sort_key, reverse=True)

    how_many = min(num_sentences, len(ranked_indexes))
    top_sentences = []
    for j in range(how_many):
        idx = ranked_indexes[j]
        top_sentences.append(original_for_scoring[idx])

    return top_sentences
