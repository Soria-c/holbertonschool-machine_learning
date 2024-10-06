#!/usr/bin/env python3
"""
Dataset class for machine translation with Portuguese-English tokenization.
"""

import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


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
        # Tokenize the dataset examples
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Apply the tokenization to the dataset using tf_encode
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens, including start and end tokens.

        Args:
            pt: A tf.Tensor containing the Portuguese sentence.
            en: A tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tokens: np.ndarray containing the tokenized Portuguese sentence.
            en_tokens: np.ndarray containing the tokenized English sentence.
        """
        # Get the vocabulary sizes from both tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Add start and end tokens for Portuguese
        pt_start_token = vocab_size_pt
        pt_end_token = vocab_size_pt + 1

        # Add start and end tokens for English
        en_start_token = vocab_size_en
        en_end_token = vocab_size_en + 1

        # Tokenize the sentences
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'),
            add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'),
            add_special_tokens=False)

        # Append the start and end tokens to the tokenized sentences
        pt_tokens = [pt_start_token] + pt_tokens + [pt_end_token]
        en_tokens = [en_start_token] + en_tokens + [en_end_token]

        # # Convert the tokens to NumPy arrays
        # pt_tokens = np.array(pt_tokens, dtype=np.int32)
        # en_tokens = np.array(en_tokens, dtype=np.int32)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method.

        Args:
            pt: A tf.Tensor containing the Portuguese sentence.
            en: A tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tensor: TensorFlow tensor containing the Portuguese
                    tokens with defined shape.
            en_tensor: TensorFlow tensor containing the English
                        tokens with defined shape.
        """
        # Encode the sentences using the encode method
        pt_tokens, en_tokens = self.encode(pt, en)

        # Convert them into TensorFlow tensors
        pt_tensor = tf.convert_to_tensor(pt_tokens)
        en_tensor = tf.convert_to_tensor(en_tokens)

        # Set the shape of the tensors dynamically
        pt_tensor.set_shape([None])
        en_tensor.set_shape([None])

        return pt_tensor, en_tensor
