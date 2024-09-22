#!/usr/bin/env python3
"""
Fast Text
"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Creates, builds, and trains a FastText model.

    Arguments:
    - sentences: List of sentences to be trained on (tokenized).
    - vector_size: Dimensionality of the embedding layer.
    - min_count: Minimum number of occurrences of a word for use in training.
    - window: Maximum distance between the current
      and predicted word within a sentence.
    - negative: Size of negative sampling.
    - cbow: Boolean to determine the training type;
      True for CBOW, False for Skip-gram.
    - epochs: Number of iterations to train over.
    - seed: Seed for the random number generator.
    - workers: Number of worker threads to train the model.

    Returns:
    - model: The trained FastText model.
    """

    # Create the FastText model
    model = gensim.models.FastText(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        # CBOW if cbow=True (sg=0), Skip-gram if cbow=False (sg=1)
        sg=0 if cbow else 1,
        negative=negative,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    model.build_vocab(sentences)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
