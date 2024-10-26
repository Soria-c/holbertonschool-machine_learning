


![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)41\.82%# Transformer Applications

* Master
* By: Alexa Orrico, Software Engineer at Holberton School
* Weight: 4
* Your score will be updated as you progress.
* **Manual QA review must be done** (request it when you are done with the project)




* [Description](#description)





[Go to tasks](#)

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/9/2b6bbd4de27e8b9b147fb397906ee5e822fe6fa3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20241026%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20241026T033620Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0eb0e76935e3c46d7d0f63e3ca35ee9e79f8a3b1b67c28f37c4dabf44a9b60d9)


## Resources


**Read or watch:**


* [How machines Read](/rltoken/J0laeFn5gIsHEh2hhP48Fw "How machines Read")
* [Sub\-word tokenizers](/rltoken/luuEHmn8yqarthPbrei8KA "Sub-word tokenizers")
* [Summary of the tokenizers](/rltoken/aBx_d0p9xhcyYdo6cjP0sw "Summary of the tokenizers")
* [Subword Tokenization](/rltoken/KgJI9YXjBOmRf9RCthqv9g "Subword Tokenization")
* [Notes on BERT tokenizer and model](/rltoken/2DH5A0_Bm5rRLYvMBe6MAA "Notes on BERT tokenizer and model")
* [What is AutoTokenizer?](/rltoken/tqHc6InG4aMGhGRGk8or8g "What is AutoTokenizer?")
* [Training a new tokenizer from an old one](/rltoken/9vVn2s-BZs-E0UgcZG1e-Q "Training a new tokenizer from an old one")
* [TFDS Overview](/rltoken/GGUi9ziJ5vLQbHqT3aPYSw "TFDS Overview")
* [How Transformers Work: A Detailed Exploration of Transformer Architecture](/rltoken/KUgtQKtHcwD8vr8cpPffnQ "How Transformers Work: A Detailed Exploration of Transformer Architecture")


**References:**


* [tfds](/rltoken/IZRkjPgMMT38dzSYJzhWvQ "tfds")
    + [tfds.load](/rltoken/K0ilcVeOihLDheWFMZgAPQ "tfds.load")
* [AutoTokenizer](/rltoken/pkUfcNxQl2gmsMC_25Oz2w "AutoTokenizer")
* [encode](/rltoken/VpO4fhAS3q8tP4cmIiNOIA "train<em>new</em>from_iterator\" target=“_blank”>train<em>new</em>from_iterator</a></li>
  <li><a href=")
* [tf.py\_function](/rltoken/P8RV_fpQFZGKHClzEAxnQg "tf.py_function")
* [TFDS Keras Example](/rltoken/0brH315xMXBHt-Ni8aJQhA "TFDS Keras Example")
* [tf.linalg.band\_part](/rltoken/qrowNd_CSj2TAuGSGUUX5A "tf.linalg.band_part")
* [Customizing what happens in fit](/rltoken/exVwAMC08C91jNYHYp_JSg "Customizing what happens in fit")


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/tSsTJf_8csoUSSTXyleseg "explain to anyone"), **without the help of Google**:


### General


* How to use Transformers for Machine Translation
* How to write a custom train/test loop in Keras
* How to use Tensorflow Datasets


## Requirements


### General


* Allowed editors: `vi`, `vim`, `emacs`
* All your files will be interpreted/compiled on Ubuntu 20\.04 LTS using `python3` (version 3\.9\)
* Your files will be executed with `numpy` (version 1\.25\.2\) and `tensorflow` (version 2\.15\)
* All your files should end with a new line
* The first line of all your files should be exactly `#!/usr/bin/env python3`
* All of your files must be executable
* A `README.md` file, at the root of the folder of the project, is mandatory
* Your code should follow the `pycodestyle` style (version 2\.11\.1\)
* All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
* All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
* All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
* Unless otherwise stated, you cannot import any module except `import transformers` , `import tensorflow_datasets as tfds` and also `import tensorflow as tf` for some other tasks.


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




 mandatory
 











 Score: 100\.00% (Checks completed: 100\.00%)
 


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






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/transformer_apps`
* File: `0-dataset.py`










 Help
 




×
#### Students who are done with "0\. Dataset"

















 Review your work
 




×
#### Correction of "0\. Dataset"






**Congratulations!** All tests passed successfully!  
You are ready for your next mission!
Start a new test
Close











---

#### Result:

* README.md file exists and is not empty
* File exists
* First line contains `#!/usr/bin/env python3`
* Not allowed to import anything except `import tensorflow_datasets as tfds` and `import transformers`
* Correct output: Datasets correctly loaded
* Output check: tokenizers correctly created
* Output check: tokenizer attributes are correct
* pycodestyle validation
* Everything is documented











 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 0\. Dataset












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**6**
correction requests
  

**17/17** 
pts









### 1\. Encode Tokens




 mandatory
 






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






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/transformer_apps`
* File: `1-dataset.py`










 Help
 




×
#### Students who are done with "1\. Encode Tokens"

















 QA Review
 




×
#### 1\. Encode Tokens












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**0/7** 
pts









### 2\. TF Encode




 mandatory
 











 Score: 28\.57% (Checks completed: 28\.57%)
 


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






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/transformer_apps`
* File: `2-dataset.py`










 Help
 




×
#### Students who are done with "2\. TF Encode"

















 Review your work
 




×
#### Correction of "2\. TF Encode"






Some checks are failing. Make sure you fix them before starting a new review  
**You got this!**
Start a new test
Close











---

#### Result:

* File exists
* First line contains `#!/usr/bin/env python3`
* Not allowed to import anything except `import tensorflow_datasets as tfds` , `import transformers` and `import tensorflow as tf`
* Output Check: Tensorflow encoding correct






    + ```
    [copy_files] Filed copied: 0-main.py
      [compare] Command to run:
      ./0-main.py 2>/dev/null
      su student_jail -c 'timeout 60 bash -c '"'"'./0-main.py 2>/dev/null'"'"''
      [compare] Return code: 1
      [compare] Student stdout:
      [compare] Student stdout length: 6
      [compare] Student stderr:
      [compare] Student stderr length: 0
      [compare] Desired stdout:
      True
      True
      <dtype: 'int64'>
      <dtype: 'int64'>
      True
      <dtype: 'int64'>
      <dtype: 'int64'>
      [compare] Desired stdout length: 89
      
    ```
* pycodestyle validation
* Everything is documented











 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 2\. TF Encode












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**8**
correction requests
  

**2/7** 
pts









### 3\. Pipeline




 mandatory
 











 Score: 16\.67% (Checks completed: 16\.67%)
 


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

*Note : The output may vary*







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/transformer_apps`
* File: `3-dataset.py`










 Help
 




×
#### Students who are done with "3\. Pipeline"

















 Review your work
 




×
#### Correction of "3\. Pipeline"






Some checks are failing. Make sure you fix them before starting a new review  
**You got this!**
Start a new test
Close











---

#### Result:

* File exists
* First line contains `#!/usr/bin/env python3`
* Not allowed to import anything except `import tensorflow_datasets as tfds` , `import transformers` and `import tensorflow as tf`
* Output Check : Filter sentences






    + ```
    [copy_files] Filed copied: 0-main.py
      [compare] Command to run:
      ./0-main.py 2>/dev/null
      su student_jail -c 'timeout 3000 bash -c '"'"'./0-main.py 2>/dev/null'"'"''
      [compare] Return code: 1
      [compare] Student stdout:
      [compare] Student stdout length: 0
      [compare] Student stderr:
      [compare] Student stderr length: 0
      [compare] Desired stdout:
      Train : pt sentence shape is valid.
      Train : en sentence shape is valid.
      Validation : pt sentence shape is valid.
      Validation : en sentence shape is valid.
      [compare] Desired stdout length: 160
      
    ```
* Output Check : Padded batches and Prefetch
* pycodestyle validation
* Everything is documented











 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 3\. Pipeline












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**1**
correction request
  

**2/12** 
pts









### 4\. Create Masks




 mandatory
 











 Score: 28\.57% (Checks completed: 28\.57%)
 


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






**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/transformer_apps`
* File: `4-create_masks.py`










 Help
 




×
#### Students who are done with "4\. Create Masks"

















 Review your work
 




×
#### Correction of "4\. Create Masks"






Some checks are failing. Make sure you fix them before starting a new review  
**You got this!**
Start a new test
Close











---

#### Result:

* File exists
* First line contains `#!/usr/bin/env python3`
* Not allowed to import anything except `import tensorflow as tf`
* Correct output: Normal






    + ```
    [copy_files] Filed copied: 0-main.py
      [compare] Command to run:
      ./0-main.py 2>/dev/null
      su student_jail -c 'timeout 600 bash -c '"'"'./0-main.py 2>/dev/null'"'"''
      [compare] Return code: 1
      [compare] Student stdout:
      [compare] Student stdout length: 0
      [compare] Student stderr:
      [compare] Student stderr length: 0
      [compare] Desired stdout:
      [[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 1. 1. 1. 1. 1. 1. 1.]]]
       [[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
          1. 1. 1. 1. 1. 1. 1. 1.]]]]
      [[[[0. 1. 1. ... 1. 1. 1.]
         [0. 0. 1. ... 1. 1. 1.]
         [0. 0. 0. ... 1. 1. 1.]
         ...
         [0. 0. 0. ... 0. 1. 1.]
         [0. 0. 0. ... 0. 1. 1.]
         [0. 0. 0. ... 0. 1. 1.]]]
       [[[0. 1. 1. ... 1. 1. 1.]
         [0. 0. 1. ... 1. 1. 1.]
         [0. 0. 0. ... 1. 1. 1.]
         ...
         [0. 0. 0. ... 1. 1. 1.]
         [0. 0. 0. ... 1. 1. 1.]
         [0. 0. 0. ... 1. 1. 1.]]]]
      [[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 1. 1. 1. 1. 1. 1. 1.]]]
       [[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
          1. 1. 1. 1. 1. 1. 1. 1.]]]]
      [compare] Desired stdout length: 755
      
    ```
* pycodestyle validation
* Everything is documented











 Requirement success
 


 Requirement fail
 




 Code success
 


 Code fail
 




 Efficiency success
 


 Efficiency fail
 




 Text answer success
 


 Text answer fail
 




 Skipped \- Previous check failed
 








 QA Review
 




×
#### 4\. Create Masks












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**1**
correction request
  

**2/7** 
pts









### 5\. Train




 mandatory
 






Take your implementation of a transformer from our [previous project](/rltoken/k7W6aOHrVXbc07p1CIC7CA "previous project") and save it to the file `5-transformer.py`. Note, you may need to make slight adjustments to this model to get it to functionally train.


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







**Repo:**


* GitHub repository: `holbertonschool-machine_learning`
* Directory: `supervised_learning/transformer_apps`
* File: `5-transformer.py, 5-train.py`










 Help
 




×
#### Students who are done with "5\. Train"

















 QA Review
 




×
#### 5\. Train












##### Commit used:


* **User:**  \-\-\-
* **URL:** Click here
* **ID:** `---`
* **Author:** \-\-\-
* **Subject:** *\-\-\-*
* **Date:** \-\-\-














**0/5** 
pts









[Previous project](/projects/2338)  



 Next project 







### Score

![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)41\.82%Still some tasks to work on!

Now that you are ready to be reviewed, share your link to your peers. You can find some [here](#available-reviewers-modal).

×#### Contact one of your peers

https://intranet.hbtn.io/corrections/853590/correctDon't forget to [review one of them](/corrections/to_review).

Next project: QA Bot

[Open the next project](/projects/2361)


### Score

![Project badge](/assets/pathway/003_color-0c5b38973bfe29fff8dd86f65a213ea2d2499a7d0d9e4549f440c50dc84c9618.png)41\.82%Still some tasks to work on!

Now that you are ready to be reviewed, share your link to your peers. You can find some [here](#available-reviewers-modal).

×#### Contact one of your peers

https://intranet.hbtn.io/corrections/853590/correctDon't forget to [review one of them](/corrections/to_review).

Next project: QA Bot

[Open the next project](/projects/2361)



### Tasks list




* [Mandatory](#mandatory)
* [Advanced](#advanced)





0\. `Dataset`
**100\.00%**


1\. `Encode Tokens`



2\. `TF Encode`
**28\.57%**


3\. `Pipeline`
**16\.67%**


4\. `Create Masks`
**28\.57%**


5\. `Train`










×#### Recommended Sandboxes

New sandbox * 
* US East (N. Virginia)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* South America (São Paulo)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* Europe (Paris)
* Ubuntu 18\.04
* Ubuntu 22\.04
* 
* Asia Pacific (Sydney)
* Ubuntu 18\.04
* Ubuntu 22\.04
No sandboxes yet!





