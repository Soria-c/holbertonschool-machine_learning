#!/usr/bin/env python3
"""
Bag of Words
"""
import numpy as np
from collections import Counter


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

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

    # Tokenize sentences into words
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]

    # If no vocabulary is provided, build vocab from all words in the sentences
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences
                           for word in sentence))

    # Create a mapping from words to feature indices
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    # Initialize the embeddings matrix (s sentences, f features/words in vocab)
    embeddings = np.zeros((len(sentences), len(vocab)))

    # Populate the embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        # Count occurrences of words in the sentence
        word_count = Counter(sentence)
        for word, count in word_count.items():
            if word in vocab_dict:
                # Fill in word frequency
                embeddings[i, vocab_dict[word]] = count

    return embeddings, vocab
