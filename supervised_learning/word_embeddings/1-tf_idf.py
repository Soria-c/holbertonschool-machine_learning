#!/usr/bin/env python3
"""
TF IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix using TfidfVectorizer.

    Arguments:
    - sentences: List of sentences to analyze.
    - vocab: List of the vocabulary words to use for the analysis.
    If None, use all words in sentences.

    Returns:
    - embeddings: numpy.ndarray of shape (s, f) containing the embeddings
                  where s is the number of sentences
                  and f is the number of features.
    - features: List of the features used for embeddings (words).
    """

    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to get the TF-IDF matrix
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Get the features (words)
    features = vectorizer.get_feature_names_out()

    return embeddings, features
