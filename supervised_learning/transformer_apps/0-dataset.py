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

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
                self.data_train
            )

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
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Load pre-trained tokenizer models from Hugging Face transformers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            clean_up_tokenization_spaces=True,
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            clean_up_tokenization_spaces=True,
            use_fast=True
        )

        # Train both tokenizers (adjust if necessary)
        # Tokenizers have already been pre-trained, no need for
        # additional vocabulary
        # training, hence no need to call any 'train' method.
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en
