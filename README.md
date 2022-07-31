# Georgian Language Modelling
We have trained popular NLP models on Georgian Language and compared the results on text generation task.

This repository contains not only the inference part of the models, but also the training code.

 We trained N-Gram,Word2Vec, LSTM and BERT models on the Georgian corpus.

## Setting up the environment
In order to use everything in this repository, you need to install the following dependencies:
```
transformers==4.12.3
torch==1.12.0
tensorflow==2.3.0
seaborn==0.11.1
nltk==3.6.1
datasets==2.4.0
gensim==4.2.0
```
For installing that dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```
## Data
For this project, we used OSCAR corpus. The corpus is available at [OSCAR](https://huggingface.co/datasets/oscar). You can download the corpus by using the following code:

```python
from datasets import load_dataset
georgian_oscar = load_dataset("oscar", "unshuffled_original_ka", split="train")
text = georgian_oscar['text']
```
You can see the preprocessing and EDA of the corpus in the following notebook:
[Preprocessing](./notebooks/EDA_Pre_Processing.ipynb)


## Structure of The Repository
The repository contains the following notebooks:
- [Preprocessing](./notebooks/EDA_Pre_Processing.ipynb) - Preprocessing of the corpus
- [N Gram Training](./notebooks/n_gram.ipynb) - Training of N-Gram model
- [Word2Vec Training](./notebooks/word2vec_training.ipynb) - Training of Word2Vec model
- [Word2Vec Analysis](./notebooks/word2vec_analysis.ipynb) - Training of Word2Vec model
- [LSTM Training](./notebooks/lstm.ipynb) - Training of LSTM model
- [BERT Training](./notebooks/trainbert.ipynb) - Training of BERT model
- [Inference](./notebooks/inference.ipynb) - Inference of the BERT model, including beam search and greedy search


## Appendix
In order to see the detailed analysis and the report of the whole project you can read [Report](./report.pdf)

If you want to use our pretrained model weights you can download the weights from [here](https://drive.google.com/drive/folders/1SI3SE5ZezXcboe5U-GdJIh9PAsKE99SU).

People Working on the project:
- [Shota Elkanishvili](https://github.com/sHOTa-23) 
- [Otar Emnadze](https://github.com/Oemnadze) 