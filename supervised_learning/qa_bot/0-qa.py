#!/usr/bin/env python3
"""
This function finds a snippet of text
within a reference document to answer a question.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Load the BERT model from TensorFlow Hub and tokenizer from Hugging Face
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")


def question_answer(question, reference):
    """
    Finds an answer to a question within a reference text.

    Parameters:
    question (str): The question to answer.
    reference (str): The reference document in which to search for the answer.

    Returns:
    str: A snippet containing the answer, or None if no answer is found.
    """
    # Tokenize input question and reference text
    inputs = tokenizer.encode_plus(question, reference,
                                   add_special_tokens=True,
                                   return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]

    # Get model predictions for start and end token positions
    outputs = model(inputs)
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]

    # Find the most probable start and end positions of the answer
    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

    # If the end position is before the start, return None
    if start_index >= end_index:
        return None

    # Decode the tokenized answer
    answer_tokens = input_ids[start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer if answer else None
