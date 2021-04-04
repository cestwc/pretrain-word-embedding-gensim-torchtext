# pretrain-word-embedding-gensim-torchtext
This is a helper to pretrain word embeddings (using gensim API) for data with format compatible with Torchtext

So now we assume that you have a customized dataset compatible to Torchtext. You may get yourself familiar with how this dataset look like by looking at this [Tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb).

You may also want to load a pretrained word embedding in the way it is done [here](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb). Certainly, you can choose to use standard GloVe word vectors from Torchtext, but that only works if you set your embedding dimension to be fixed at 25, 50, 100, 200, or 300. What if you would like to use 128, for example, as your embedding dimension? (Using 128 is actually quite common in many works.)

[Gensim](https://github.com/RaRe-Technologies/gensim-data) could be a good option to select a corpus, using which you can pretrain your word embeddings, or a good option to select a pretrained word embedding model (apart from GloVe) that later loaded into your subsequent model. Word embedding pretraining using [Gensim](https://radimrehurek.com/gensim/models/fasttext.html) usually looks like this:

```python
from gensim.models import FastText

common_texts = [['human', 'interface', 'computer'], ['human',  'beings']] # a list of tokenized sentences
model = FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10) # keyword 'iter' may be 'epochs, 'size' may be 'vector_size', depending on versions
```

Here, we just use this API and feed it with our dataset that is not yet in the form of "a list of tokenized sentences", but one (or more) ```json``` file(s), instead. These ```json``` files are exactly the ones you use for your model learning with PyTorch / Torchtext.

Since you certainly don't want to pretrain your embeddings every time you run the entire code, a better compromise would be saving the vectors as ```.txt``` and load it later in Torchtext.
