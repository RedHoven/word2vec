There exist various implementations of word2vec. As I intend to train it on a small corpus, I will implement a skip-gram with negative sampling (SGNS) model.

## Math derivation for the implementation

The goal of the model is to increase the probability of the context words given the center word. The probability is defined as follows:

```math
P(w_o | w_c) =\frac{\exp(v_{w_o} \cdot u_{w_c})}{\sum_{w \in V} \exp(v_w \cdot u_{w_c})}
```

Here:
- $u_w$ is the center embedding of word $w$
- $v_w$ is the context embedding of word $w$
- $V$ is the vocabulary

To maximize this probability (via Maximum Likelihood Estimation), we minimize the negative log-likelihood (cross-entropy loss):

```math
L = -\sum_{(w_c, w_o) \in D} \log P(w_o | w_c)
```

where $D$ is the dataset of center–context pairs.

However, computing the denominator of the softmax requires iterating over the entire vocabulary, which is computationally expensive. To address this, we can use negative sampling, which approximates the objective function by turning the problem into a set of binary classification tasks. For each observed pair $w_c, w_o$, we sample $K$ negative words $w_{n_1}, \dots, w_{n_K}$. The loss for a single training pair becomes:

```math
L =-\log \sigma(v_p \cdot u)-\sum_{k=1}^{K} \log \sigma(-v_{n_k} \cdot u)
```

where:
* $u$ is the center word embedding
* $v_p$ is the embedding of the positive context word
* $v_{n_k}$ are the embeddings of the negative samples
* $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function

This objective encourages the model to assign high similarity to true context pairs and low similarity to randomly sampled word pairs.

Since the model is linear (no hidden non-linearities), we can directly compute the gradients of the loss with respect to the embeddings. Gradient for the center embedding $u$ is computed as follows:

```math
\frac{\partial L}{\partial u}=\frac{\partial}{\partial u}\left(-\log \sigma(v_p \cdot u)-\sum_{k=1}^{K} \log \sigma(-v_{n_k} \cdot u)\right)
```

Taking derivatives term-by-term:

```math
\frac{\partial}{\partial u}[-\log \sigma(v_p \cdot u)]
=
(\sigma(v_p \cdot u) - 1) v_p
```

and

```math
\frac{\partial}{\partial u}[-\log \sigma(-v_{n_k} \cdot u)]
=
\sigma(v_{n_k} \cdot u) v_{n_k}
```

Therefore:

```math
\boxed{
\frac{\partial L}{\partial u}
=
(\sigma(v_p \cdot u)-1) v_p
+
\sum_{k=1}^{K} \sigma(v_{n_k} \cdot u) v_{n_k}
}
```

Gradient with respect to the positive context embedding is:

```math
\frac{\partial L}{\partial v_p}
=
\frac{\partial}{\partial v_p}[-\log \sigma(v_p \cdot u)]
```

which gives

```math
\boxed{
\frac{\partial L}{\partial v_p}
=
(\sigma(v_p \cdot u)-1) u
}
```

Gradient with respect to the negative context embeddings is:

```math
\frac{\partial L}{\partial v_{n_k}}
=
\frac{\partial}{\partial v_{n_k}}[-\log \sigma(-v_{n_k} \cdot u)]
```

which simplifies to

```math
\boxed{
\frac{\partial L}{\partial v_{n_k}}
=
\sigma(v_{n_k} \cdot u) u
}
```

Using stochastic gradient descent, the embeddings are updated as:

```math
\theta_{new} = \theta_{old} - \alpha \nabla_\theta L
```

where:

* $\alpha$ is the learning rate, scheduled to decrease over time
* $\theta$ represents the parameters ($u$, $v_p$, $v_{n_k}$)

## Sources and inspiration for the implementation:

- Mikolov, Tomas, et al. ["Efficient estimation of word representations in vector space."](https://arxiv.org/abs/1301.3781) *arXiv preprint arXiv:1301.3781* (2013).
- Mikolov, Tomas, et al. ["Distributed representations of words and phrases and their compositionality."](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) *Advances in Neural Information Processing Systems*, 26 (2013).
- [Notes for Stanford CS224N: NLP with Deep Learning](https://github.com/jaaack-wang/Notes-for-Stanford-CS224N-NLP-with-Deep-Learning/tree/55d1fdf7800d1ee0833586f3d15de092aed4ae8d)
- [Dive into Deep Learning: Word2Vec](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)
- [Word2Vec Tutorial: The Skip-Gram Model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
