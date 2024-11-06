# Natural Language Processing \- Evaluation Metrics


## Resources


**Read or watch:**


* [7 Applications of Deep Learning for Natural Language Processing](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/ "7 Applications of Deep Learning for Natural Language Processing")
* [10 Applications of Artificial Neural Networks in Natural Language Processing](https://medium.com/product-ai/artificial-neural-networks-in-natural-language-processing-bcf62aa9151a "10 Applications of Artificial Neural Networks in Natural Language Processing")
* [A Gentle Introduction to Calculating the BLEU Score for Text in Python](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/ "A Gentle Introduction to Calculating the BLEU Score for Text in Python")
* [Bleu Score](https://www.youtube.com/watch?v=DejHQYAGb7Q "Bleu Score")
* [Evaluating Text Output in NLP: BLEU at your own risk](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213 "Evaluating Text Output in NLP: BLEU at your own risk")
* [ROUGE metric](https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460 "ROUGE metric")
* [Evaluation and Perplexity](https://www.youtube.com/watch?v=BAN3NB_SNHY "Evaluation and Perplexity")
* [Evaluation metrics](https://aman.ai/primers/ai/evaluation-metrics/ "Evaluation metrics")


**Definitions to skim**


* [BLEU](https://en.wikipedia.org/wiki/BLEU "BLEU")
* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric) "ROUGE")
* [Perplexity](https://en.wikipedia.org/wiki/Perplexity "Perplexity")


**References:**


* [BLEU: a Method for Automatic Evaluation of Machine Translation (2002\)](https://aclanthology.org/P02-1040.pdf "BLEU: a Method for Automatic Evaluation of Machine Translation (2002)")
* [ROUGE: A Package for Automatic Evaluation of Summaries (2004\)](https://aclanthology.org/W04-1013.pdf` "ROUGE: A Package for Automatic Evaluation of Summaries (2004)")


## Learning Objectives


At the end of this project, you are expected to be able to [explain to anyone](/rltoken/JiW-QDEd5m2d5PzoZiH0rw "explain to anyone"), **without the help of Google**:


### General


* What are the applications of natural language processing?
* What is a BLEU score?
* What is a ROUGE score?
* What is perplexity?
* When should you use one evaluation metric over another?


## Requirements


* You are not allowed to use the `nltk` module






#### Question \#0



The BLEU score measures:



* A model’s accuracy
* ***A model’s precision***
* A model’s recall
* A model’s perplexity







#### Question \#1



The ROUGE score measures:



* A model’s accuracy
* ***A model’s precision***
* ***A model’s recall***
* A model’s perplexity







#### Question \#2



Perplexity measures:



* The accuracy of a prediction
* ***The branching factor of a prediction***
* A prediction’s recall
* A prediction’s accuracy







#### Question \#3



The BLEU score was designed for:



* Sentiment Analysis
* ***Machine Translation***
* Question\-Answering
* Document Summarization







#### Question \#4



What are the shortcomings of the BLEU score?



* ***It cannot judge grammatical accuracy***
* ***It cannot judge meaning***
* ***It does not work with languages that lack word boundaries***
* ***A higher score is not necessarily indicative of a better translation***











## Tasks







### 0\. Unigram BLEU score




Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:


* `references` is a list of reference translations
    + each reference translation is a list of the words in the translation
* `sentence` is a list containing the model proposed sentence
* Returns: the unigram BLEU score



```
$ cat 0-main.py
  #!/usr/bin/env python3
  
  uni_bleu = __import__('0-uni_bleu').uni_bleu
  
  references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
  sentence = ["there", "is", "a", "cat", "here"]
  
  print(uni_bleu(references, sentence))
  $ ./0-main.py
  0.6549846024623855
  $
  
```



### 1\. N\-gram BLEU score




Write the function `def ngram_bleu(references, sentence, n):` that calculates the n\-gram BLEU score for a sentence:


* `references` is a list of reference translations
    + each reference translation is a list of the words in the translation
* `sentence` is a list containing the model proposed sentence
* `n` is the size of the n\-gram to use for evaluation
* Returns: the n\-gram BLEU score



```
$ cat 1-main.py
  #!/usr/bin/env python3
  
  ngram_bleu = __import__('1-ngram_bleu').ngram_bleu
  
  references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
  sentence = ["there", "is", "a", "cat", "here"]
  
  print(ngram_bleu(references, sentence, 2))
  $ ./1-main.py
  0.6140480648084865
  $
  
```

### 2\. Cumulative N\-gram BLEU score





Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n\-gram BLEU score for a sentence:


* `references` is a list of reference translations
    + each reference translation is a list of the words in the translation
* `sentence` is a list containing the model proposed sentence
* `n` is the size of the largest n\-gram to use for evaluation
* All n\-gram scores should be weighted evenly
* Returns: the cumulative n\-gram BLEU score



```
$ cat 2-main.py
  #!/usr/bin/env python3
  
  cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu
  
  references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
  sentence = ["there", "is", "a", "cat", "here"]
  
  print(cumulative_bleu(references, sentence, 4))
  $ ./2-main.py
  0.5475182535069453
  $
  
```






