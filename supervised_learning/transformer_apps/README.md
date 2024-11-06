# Transformer Applications

## Resources


**Read or watch:**


* [How machines Read](https://www.openteams.com/tokenizers-how-machines-read/ "How machines Read")
* [Sub\-word tokenizers](https://medium.com/@harshitasharma_ai/sub-word-tokenizers-85323fb64aaf "Sub-word tokenizers")
* [Summary of the tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary "Summary of the tokenizers")
* [Subword Tokenization](https://www.thoughtvector.io/blog/subword-tokenization/ "Subword Tokenization")
* [Notes on BERT tokenizer and model](https://medium.com/@anmolkohli/my-notes-on-bert-tokenizer-and-model-98dc22d0b64#:~:text=BERT%20tokenizer%20uses%20something%20known,algorithm%20to%20generate%20the%20vocabulary "Notes on BERT tokenizer and model")
* [What is AutoTokenizer?](https://app.dataquest.io/m/796/building-text-models-with-transformers/4/autotokenizer "What is AutoTokenizer?")
* [Training a new tokenizer from an old one](https://huggingface.co/learn/nlp-course/chapter6/2#training-a-new-tokenizer-from-an-old-one "Training a new tokenizer from an old one")
* [TFDS Overview](https://www.tensorflow.org/datasets/overview "TFDS Overview")
* [How Transformers Work: A Detailed Exploration of Transformer Architecture](https://www.datacamp.com/tutorial/how-transformers-work "How Transformers Work: A Detailed Exploration of Transformer Architecture")


**References:**


* [tfds](https://www.tensorflow.org/datasets/api_docs/python/tfds "tfds")
    + [tfds.load](https://www.tensorflow.org/datasets/api_docs/python/tfds/load "tfds.load")
* [AutoTokenizer](https://huggingface.co/docs/transformers/v4.15.0/model_doc/auto#transformers.AutoTokenizer "AutoTokenizer")
* [encode](https://huggingface.co/transformers/v4.9.2/main_classes/tokenizer.html#transformers.PreTrainedTokenizerFast.train_new_from_iterator)
* [tf.py\_function](https://www.tensorflow.org/api_docs/python/tf/py_function "tf.py_function")
* [TFDS Keras Example](https://www.tensorflow.org/datasets/keras_example "TFDS Keras Example")
* [tf.linalg.band\_part](https://www.tensorflow.org/api_docs/python/tf/linalg/band_part "tf.linalg.band_part")
* [Customizing what happens in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit "Customizing what happens in fit")


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/tSsTJf_8csoUSSTXyleseg "explain to anyone"), **without the help of Google**:


### General


* How to use Transformers for Machine Translation
* How to write a custom train/test loop in Keras
* How to use Tensorflow Datasets


## TF Datasets


For machine translation, we will be using the prepared [Tensorflow Datasets](/rltoken/bLhqBU5gOtUjjZjLc1qrWQ "Tensorflow Datasets") [ted\_hrlr\_translate/pt\_to\_en](/rltoken/bk9w62FECGXMsqdJC_qKVw "ted_hrlr_translate/pt_to_en") for English to Portuguese translation


To download Tensorflow Datasets, please use:



```
pip install --user tensorflow-datasets==4.9.2
  
```

To use this dataset:



```
$ cat load_dataset.py
  #!/usr/bin/env python3
  import tensorflow as tf
  import tensorflow_datasets as tfds
  
  pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
  for pt, en in pt2en_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
  $ ./load_dataset.py
  e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
  and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
  
```

## Transformers


To download transformers library, please use:



```
pip install --user transformers==4.44.2
  
```






## Tasks







### 0\. Dataset


Create the class `Dataset` that loads and preps a dataset for machine translation:


* Class constructor `def __init__(self):`
    + creates the instance attributes:
        - `data_train`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `train` split, loaded `as_supervided`
        - `data_valid`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `validate` split, loaded `as_supervided`
        - `tokenizer_pt` is the Portuguese tokenizer created from the training set
        - `tokenizer_en` is the English tokenizer created from the training set
* Create the instance method `def tokenize_dataset(self, data):` that creates sub\-word tokenizers for our dataset:
    + `data` is a `tf.data.Dataset` whose examples are formatted as a tuple `(pt, en)`
        - `pt` is the `tf.Tensor` containing the Portuguese sentence
        - `en` is the `tf.Tensor` containing the corresponding English sentence
    + Use a pre\-trained tokenizer:
        - use the pretrained model `neuralmind/bert-base-portuguese-cased` for the `portuguese` text
        - use the pretrained model `bert-base-uncased` for the `english` text
    + Train the tokenizers with a maximum vocabulary size of `2**13`
    + Returns: `tokenizer_pt, tokenizer_en`
        - `tokenizer_pt` is the Portuguese tokenizer
        - `tokenizer_en` is the English tokenizer



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  Dataset = __import__('0-dataset').Dataset
  
  data = Dataset()
  for pt, en in data.data_train.take(1):
      print(pt.numpy().decode('utf-8'))
      print(en.numpy().decode('utf-8'))
  for pt, en in data.data_valid.take(1):
      print(pt.numpy().decode('utf-8'))
      print(en.numpy().decode('utf-8'))
  print(type(data.tokenizer_pt))
  print(type(data.tokenizer_en))
  $ ./0-main.py
  e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
  and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
  tinham comido peixe com batatas fritas ?
  did they eat fish and chips ?
  <class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
  <class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
  $
  
```

### 1\. Encode Tokens

Update the class `Dataset`:


* Create the instance method `def encode(self, pt, en):` that encodes a translation into tokens:
    + `pt` is the `tf.Tensor` containing the Portuguese sentence
    + `en` is the `tf.Tensor` containing the corresponding English sentence
    + The tokenized sentences should include the start and end of sentence tokens
    + The start token should be indexed as `vocab_size`
    + The end token should be indexed as `vocab_size + 1`
    + Returns: `pt_tokens, en_tokens`
        - `pt_tokens` is a `np.ndarray` containing the Portuguese tokens
        - `en_tokens` is a `np.ndarray.` containing the English tokens



```
$ cat 1-main.py
  #!/usr/bin/env python3
  
  Dataset = __import__('1-dataset').Dataset
  
  data = Dataset()
  for pt, en in data.data_train.take(1):
      print(data.encode(pt, en))
  for pt, en in data.data_valid.take(1):
      print(data.encode(pt, en))
  $ ./1-main.py
  ([8192, 45, 363, 748, 262, 41, 1427, 15, 7015, 262, 41, 1499, 5524, 252, 4421, 15, 201, 84, 41, 300, 395, 693, 314, 17, 8193], [8192, 122, 282, 140, 2164, 2291, 1587, 14, 140, 391, 501, 898, 113, 240, 4451, 129, 2689, 14, 379, 145, 838, 2216, 508, 254, 16, 8193])
  ([8192, 1274, 209, 380, 4767, 209, 1937, 6859, 46, 239, 666, 33, 8193], [8192, 386, 178, 1836, 2794, 122, 5953, 31, 8193])
  $
  
```



### 2\. TF Encode

Update the class `Dataset`:


* Create the instance method `def tf_encode(self, pt, en):` that acts as a `tensorflow` wrapper for the `encode` instance method
    + Make sure to set the shape of the `pt` and `en` return tensors
* Update the class constructor `def __init__(self):`
    + update the `data_train` and `data_validate` attributes by tokenizing the examples



```
$ cat 2-main.py
  #!/usr/bin/env python3
  
  Dataset = __import__('2-dataset').Dataset
  
  data = Dataset()
  for pt, en in data.data_train.take(1):
      print(pt, en)
  for pt, en in data.data_valid.take(1):
      print(pt, en)
  $ ./2-main.py
  tf.Tensor(
  [8192   45  363  748  262   41 1427   15 7015  262   41 1499 5524  252
   4421   15  201   84   41  300  395  695  314   17 8193], shape=(25,), dtype=int64) tf.Tensor(
  [8192  122  282  140 2164 2291 1587   14  140  391  501  898  113  240
   4451  129 2689   14  379  145  838 2216  508  254   16 8193], shape=(26,), dtype=int64)
  tf.Tensor([8192 1274  209  380 4767  209 1937 6859   46  239  666   33 8193], shape=(13,), dtype=int64) tf.Tensor([8192  386  178 1836 2794  122 5953   31 8193], shape=(9,), dtype=int64)
  $
  
```

### 3\. Pipeline

Update the class `Dataset` to set up the data pipeline:


* Update the class constructor `def __init__(self, batch_size, max_len):`
    + `batch_size` is the batch size for training/validation
    + `max_len` is the maximum number of tokens allowed per example sentence
    + update the `data_train` attribute by performing the following actions:
        - filter out all examples that have either sentence with more than `max_len` tokens
        - cache the dataset to increase performance
        - shuffle the entire dataset using a `buffer size` equal to `20000`.
        - split the dataset into padded batches of size `batch_size`
        - prefetch the dataset using `tf.data.experimental.AUTOTUNE` to increase performance
    + update the `data_validate` attribute by performing the following actions:
        - filter out all examples that have either sentence with more than `max_len` tokens
        - split the dataset into padded batches of size `batch_size`



```
$ cat 3-main.py
  #!/usr/bin/env python3
  
  Dataset = __import__('3-dataset').Dataset
  import tensorflow as tf
  
  tf.random.set_seed(0)
  data = Dataset(32, 40)
  for pt, en in data.data_train.take(1):
      print(pt, en)
  for pt, en in data.data_valid.take(1):
      print(pt, en)
  $ ./3-main.py
  tf.Tensor(
  [[8192 6633   29 ...    0    0    0]
   [8192  516 5468 ...    0    0    0]
   [8192  855 1038 ...    0    0    0]
   ...
   [8192 2500  121 ...    0    0    0]
   [8192   55  201 ...    0    0    0]
   [8192  363  936 ...    0    0    0]], shape=(32, 36), dtype=int64) tf.Tensor(
  [[8192 5107   28 ...    0    0    0]
   [8192 5890 5486 ...    0    0    0]
   [8192  171  224 ...    0    0    0]
   ...
   [8192   46  315 ...    0    0    0]
   [8192  192  145 ...    0    0    0]
   [8192  282  136 ...    0    0    0]], shape=(32, 36), dtype=int64)
  tf.Tensor(
  [[8192 1274  209 ...    0    0    0]
   [8192  580  796 ...    0    0    0]
   [8192 3073  116 ...    0    0    0]
   ...
   [8192 2948   16 ...    0    0    0]
   [8192  333  981 ...    0    0    0]
   [8192  421 5548 ...    0    0    0]], shape=(32, 37), dtype=int64) tf.Tensor(
  [[8192  386  178 ...    0    0    0]
   [8192   46  176 ...    0    0    0]
   [8192   46 4783 ...    0    0    0]
   ...
   [8192   46 1132 ...    0    0    0]
   [8192  135  145 ...    0    0    0]
   [8192  122  979 ...    0    0    0]], shape=(32, 38), dtype=int64)
  $
  
```

### 4\. Create Masks


Create the function `def create_masks(inputs, target):` that creates all masks for training/validation:


* `inputs` is a tf.Tensor of shape `(batch_size, seq_len_in)` that contains the input sentence
* `target` is a tf.Tensor of shape `(batch_size, seq_len_out)` that contains the target sentence
* This function should only use `tensorflow` operations in order to properly function in the training step
* Returns: `encoder_mask`, `combined_mask`, `decoder_mask`
    + `encoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` to be applied in the encoder
    + `combined_mask` is the `tf.Tensor` of shape `(batch_size, 1, seq_len_out, seq_len_out)` used in the 1st attention block in the decoder to pad and mask future tokens in the input received by the decoder. It takes the maximum between a look*ahead*mask and the decoder target padding mask.
    + `decoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` used in the 2nd attention block in the decoder.



```
$ cat 4-main.py
  #!/usr/bin/env python3
  
  Dataset = __import__('3-dataset').Dataset
  create_masks = __import__('4-create_masks').create_masks
  import tensorflow as tf
  
  tf.random.set_seed(0)
  data = Dataset(32, 40)
  for inputs, target in data.data_train.take(1):
      print(create_masks(inputs, target))
  $ ./4-main.py
  (<tf.Tensor: shape=(32, 1, 1, 36), dtype=float32, numpy=
  array([[[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         ...,
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 1, 36, 36), dtype=float32, numpy=
  array([[[[0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           ...,
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           ...,
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           ...,
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.]]],
  
  
         ...,
  
  
         [[[0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           ...,
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           ...,
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           ...,
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: shape=(32, 1, 1, 36), dtype=float32, numpy=
  array([[[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         ...,
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]],
  
  
         [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>)
  $
  
```

### 5\. Train


Take your implementation of a transformer from our [previous project](/supervised_learning/attention/README.md "previous project") and save it to the file `5-transformer.py`. Note, you may need to make slight adjustments to this model to get it to functionally train.


Write a the function `def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):` that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:


* `N` the number of blocks in the encoder and decoder
* `dm` the dimensionality of the model
* `h` the number of heads
* `hidden` the number of hidden units in the fully connected layers
* `max_len` the maximum number of tokens per sequence
* `batch_size` the batch size for training
* `epochs` the number of epochs to train for
* You should use the following imports:
    + `Dataset = __import__('3-dataset').Dataset`
    + `create_masks = __import__('4-create_masks').create_masks`
    + `Transformer = __import__('5-transformer').Transformer`
* Your model should be trained with Adam optimization with `beta_1=0.9`, `beta_2=0.98`, `epsilon=1e-9`
    + The learning rate should be scheduled using the following equation with `warmup_steps=4000`:
    + ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/9/39ceb6fefc25283cd8ee7a3f302ae799b6051bcd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T033620Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=5dbf4b454c37ea57572bde98f587da395d75dc15d356cd88c53ae6706abb30af)
* Your model should use sparse categorical crossentropy loss, ignoring padded tokens
* Your model should print the following information about the training:
    + Every 50 batches, you should print `Epoch {Epoch number}, batch {batch_number}: loss {training_loss} accuracy {training_accuracy}`
    + Every epoch, you should print `Epoch {Epoch number}: loss {training_loss} accuracy {training_accuracy}`
* Returns the trained model



```
$ cat 5-main.py
  #!/usr/bin/env python3
  
  import tensorflow as tf
  train_transformer = __import__('5-train').train_transformer
  
  tf.random.set_seed(0)
  transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
  print(type(transformer))
  $ ./5-main.py
  Epoch 1, Batch 0: Loss 9.033271789550781, Accuracy 0.0
  Epoch 1, Batch 50: Loss 8.9612398147583, Accuracy 0.0030440413393080235
  Epoch 1, Batch 100: Loss 8.842384338378906, Accuracy 0.02672104351222515
  
  ...
  
  
  Epoch 1, Batch 900: Loss 5.393615245819092, Accuracy 0.20997539162635803
  Epoch 1, Batch 950: Loss 5.21303653717041, Accuracy 0.21961714327335358
  Epoch 1, Batch 1000: Loss 5.040729999542236, Accuracy 0.2290351837873459
  Epoch 1: Loss 5.018191337585449, Accuracy 0.2303551733493805
  Epoch 2, Batch 0: Loss 1.6565858125686646, Accuracy 0.45769229531288147
  Epoch 2, Batch 50: Loss 1.6106284856796265, Accuracy 0.4158884584903717
  Epoch 2, Batch 100: Loss 1.5795631408691406, Accuracy 0.4239685535430908
  
  ...
  
  
  Epoch 2, Batch 900: Loss 0.8802141547203064, Accuracy 0.4850142300128937
  Epoch 2, Batch 950: Loss 0.8516387343406677, Accuracy 0.4866654574871063
  Epoch 2, Batch 1000: Loss 0.8241745233535767, Accuracy 0.4883441627025604
  Epoch 2: Loss 0.8208674788475037, Accuracy 0.4885943830013275
  <class '5-transformer.Transformer'>
  $
  
```

*Note: In this example, we only train for 2 epochs since the full training takes quite a long time. If you’d like to properly train your model, you’ll have to train for 20\+ epochs*
