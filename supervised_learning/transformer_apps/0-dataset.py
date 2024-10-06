#!/usr/bin/env python3
"""
Dataset class for machine translation with Portuguese-English tokenization.
"""

import transformers
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class"""

    def __init__(self):
        """Initializes the Dataset object with training,
        validation, and tokenizers."""
        # Load the TED Talks translation dataset
        # for Portuguese to English
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)

        # Tokenizers (will be initialized in tokenize_dataset method)
        self.tokenizer_pt = None
        self.tokenizer_en = None

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers
        for Portuguese and English.

        Args:
            data: A tf.data.Dataset where examples are tuples (pt, en)
                  pt is a Tensor with Portuguese text, en is a Tensor
                  with English text.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        # Load pre-trained tokenizer models from Hugging Face transformers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        # Train both tokenizers (adjust if necessary)
        # Tokenizers have already been pre-trained, no need for
        # additional vocabulary
        # training, hence no need to call any 'train' method.

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en
