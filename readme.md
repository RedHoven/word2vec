# Word2Vec: Skip-Gram with Negative Sampling on Simple English Wikipedia dataset

Implementation of the Word2Vec algorithm using the Skip-gram model with Negative Sampling. This code trains word embeddings on a given text corpus and allows for interactive exploration of similar words based on the learned embeddings. The dataset was taken from [Simple English Wikipedia (Kaggle)](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish).

The trained embeddings and training logs are saved in the `artifacts` directory.

`word2vec.py` - main implementation of the Word2Vec algorithm, including data preprocessing, training loop, and interactive exploration of similar words.
`math.md` - mathematical derivation of the Skip-gram with Negative Sampling loss function and its gradients.
`result.ipynb` - notebook containing visualizations of the loss and resulting embeddings using SVD.

### How to run?
- download the dataset and artifacts (if needed) from [Google Drive](https://drive.google.com/drive/folders/1veEIrsAWlAyvDHxM7fwnrti6dpqewteT?usp=sharing)
- install dependencies: `pip install -r requirements.txt`
- run training from scratch: `python word2vec.py --train`
- run interactive mode: `python word2vec.py --interactive`