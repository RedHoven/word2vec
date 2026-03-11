# Pure NumPy implementation of Skip-Gram with Negative Sampling (SGNS)
# inspired by Mikolov et al. (2013). 
# Optimizations include batch processing and learning rate scheduling for improved training efficiency and convergence.
#
# - "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)
# - "Distributed Representations of Words and Phrases and their Compositionality" (Mikolov et al., 2013)

import time
import os
import numpy as np
import re
from collections import Counter

# Hyperparameters
EMBEDDING_DIM = 100
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 20
EPOCHS = 5
INITIAL_LR = 0.025
MIN_COUNT = 20
SUBSAMPLING_THRESHOLD = 1e-5
BATCH_SIZE = 512

def sigmoid(x):
    """Sigmoid function with numerical stability"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def load_and_tokenize(filename):
    """Load data and extract tokens"""
    print("Loading data...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    tokens = re.findall(r'\b[a-z]+\b', text)
    print(f"Total tokens: {len(tokens)}")
    return tokens

def build_vocabulary(tokens, min_count):
    """Build vocabulary and count word frequencies"""
    print("Building vocabulary...")
    word_counts = Counter(tokens)

    vocab = {word: count for word, count in word_counts.items() if count >= min_count}
    
    word2idx = {word: idx for idx, word in enumerate(vocab.keys())}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, word2idx, idx2word

def subsample_tokens(tokens, word2idx, vocab, threshold):
    """Subsample frequent words"""
    print("Subsampling frequent words...")
    total_count = sum(vocab.values())
    
    subsampled = []
    for token in tokens:
        if token not in word2idx:
            continue
        
        freq = vocab[token] / total_count
        keep_prob = min(1.0, np.sqrt(threshold / freq))
        
        if np.random.random() < keep_prob:
            subsampled.append(token)
    
    print(f"Tokens after subsampling: {len(subsampled)}")
    return subsampled

def get_learning_rate(initial_lr, current_step, total_steps, min_lr_factor=0.0001):
    """
    Compute learning rate with linear decay schedule.
    """
    progress = current_step / total_steps
    lr = initial_lr * (1 - progress)
    return max(lr, initial_lr * min_lr_factor)

def sample_training_pair(token_indices, window_size):
    """
    Sample a random (center word, context word) training pair.
    """
    # Sample a random center word position
    token_idx = np.random.randint(window_size, len(token_indices) - window_size)
    center_word_idx = token_indices[token_idx]
    
    # Sample a random window size (between 1 and window_size)
    dynamic_window = np.random.randint(1, window_size + 1)
    
    # Sample one context position from the dynamic window
    context_positions = list(range(token_idx - dynamic_window, token_idx)) + \
                        list(range(token_idx + 1, token_idx + dynamic_window + 1))

    context_pos = np.random.choice(context_positions)
    context_word_idx = token_indices[context_pos]

    return center_word_idx, context_word_idx

def sample_negative_examples(neg_sampling_table, num_samples, exclude_indices=None):
    """
    Sample negative examples from the negative sampling table.
    Filters out any indices that should be excluded.
    """
    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)
    
    negatives = []
    while len(negatives) < num_samples:
        candidate = neg_sampling_table[np.random.randint(0, len(neg_sampling_table))]
        if candidate not in exclude_indices:
            negatives.append(candidate)
    
    return np.array(negatives, dtype=np.int32)

def sample_training_batch(token_indices, window_size, batch_size, neg_sampling_table, num_negatives):
    """
    Sample a batch of training examples.
    """
    center_indices = np.zeros(batch_size, dtype=np.int32)
    context_indices = np.zeros(batch_size, dtype=np.int32)
    negative_indices = np.zeros((batch_size, num_negatives), dtype=np.int32)
    
    for i in range(batch_size):
        center_indices[i], context_indices[i] = sample_training_pair(token_indices, window_size)
        exclude_indices = [center_indices[i], context_indices[i]]
        negative_indices[i] = sample_negative_examples(neg_sampling_table, num_negatives, exclude_indices=exclude_indices)
    
    # Dims:
    # center_indices: (batch_size,)
    # context_indices: (batch_size,)
    # negative_indices: (batch_size, num_negatives)
    return center_indices, context_indices, negative_indices

def compute_batch_loss_and_gradients(U, V, center_indices, context_indices, negative_indices):
    """
    Compute loss and gradients for a batch of training examples.
    """    
    u = U[center_indices]           # (batch_size, embedding_dim)
    v_pos = V[context_indices]      # (batch_size, embedding_dim)
    v_neg = V[negative_indices]     # (batch_size, num_negatives, embedding_dim)
    
    # (batch_size,)
    pos_scores = np.sum(v_pos * u, axis=1) 
    
    # (batch_size, num_negatives)
    neg_scores = np.sum(v_neg * u[:, np.newaxis, :], axis=2)
    
    pos_sig = sigmoid(pos_scores)
    pos_loss = -np.sum(np.log(pos_sig + 1e-10))
    
    neg_sig = sigmoid(neg_scores)
    neg_loss = -np.sum(np.log(1 - neg_sig + 1e-10))

    total_loss = (pos_loss + neg_loss) / BATCH_SIZE
    
    # Gradient for center embeddings (batch_size, embedding_dim)
    pos_grad_factor = (pos_sig - 1)[:, np.newaxis]
    grad_U = pos_grad_factor * v_pos
    neg_grad_factor = neg_sig[:, :, np.newaxis]
    grad_U += np.sum(neg_grad_factor * v_neg, axis=1)
    
    # Gradient for positive context embeddings (batch_size, embedding_dim)
    grad_V_pos = pos_grad_factor * u
    
    # Gradient for negative context embeddings (batch_size, num_negatives, embedding_dim)
    grad_V_neg = neg_grad_factor * u[:, np.newaxis, :]

    return total_loss, grad_U, grad_V_pos, grad_V_neg, center_indices, context_indices, negative_indices

def update_embeddings_batch(U, V, center_indices, context_indices, negative_indices, 
                           grad_U, grad_V_pos, grad_V_neg, lr):
    """
    Update embeddings using gradients from a batch of examples.
    Uses np.add.at to handle duplicate indices correctly.
    """
    # Update center and positive context embeddings
    np.add.at(U, center_indices, -lr * grad_U)
    np.add.at(V, context_indices, -lr * grad_V_pos)
    
    # Update negative embeddings
    # To make the shapes compatible with np.add.at, we need to flatten the negative indices and gradients
    # (batch_size, num_negatives) -> (batch_size * num_negatives,)
    neg_indices_flat = negative_indices.reshape(-1)
    # (batch_size, num_negatives, embedding_dim) -> (batch_size * num_negatives, embedding_dim)
    grad_V_neg_flat = grad_V_neg.reshape(-1, grad_V_neg.shape[-1])
    np.add.at(V, neg_indices_flat, -lr * grad_V_neg_flat)

def build_negative_sampling_table(vocab, table_size=int(1e7)):
    """Build negative sampling table for efficient sampling"""
    print("Building negative sampling table...")
    words = list(vocab.keys())
    counts = np.array([vocab[w] for w in words])
    
    # Raise to 3/4 power
    powered_counts = counts ** 0.75
    total = powered_counts.sum()

    table = np.zeros(table_size, dtype=np.int32)
    word_idx = 0
    cumulative_prob = powered_counts[0] / total
    
    # Fill the table with word indices according to their adjusted frequencies
    for table_idx in range(table_size):
        table[table_idx] = word_idx
        if table_idx / table_size > cumulative_prob and word_idx < len(words) - 1:
            word_idx += 1
            cumulative_prob += powered_counts[word_idx] / total
    
    return table

def train_word2vec(filename):
    """Main training function"""
    
    tokens = load_and_tokenize(filename)
    vocab, word2idx, idx2word = build_vocabulary(tokens, MIN_COUNT)
    vocab_size = len(vocab)
    
    tokens = subsample_tokens(tokens, word2idx, vocab, SUBSAMPLING_THRESHOLD)
    token_indices = [word2idx[token] for token in tokens]
    neg_sampling_table = build_negative_sampling_table(vocab)
    
    print("Initializing embeddings...")
    U = (np.random.rand(vocab_size, EMBEDDING_DIM) - 0.5) / np.sqrt(EMBEDDING_DIM)
    V = (np.random.rand(vocab_size, EMBEDDING_DIM) - 0.5) / np.sqrt(EMBEDDING_DIM)
    
    print("Training...")
    num_batches = len(token_indices) // BATCH_SIZE
    total_steps = num_batches * EPOCHS

    os.makedirs("artifacts", exist_ok=True)
    log_filename = f"artifacts/training_log_{int(time.time())}.csv"
    log_file = open(log_filename, 'w')
    log_file.write("epoch,batch,total_batches,loss,avg_loss,learning_rate\n")
    print(f"Logging training metrics to {log_filename}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        total_loss = 0
        
        for batch_idx in range(num_batches):
            center_indices, context_indices, negative_indices = sample_training_batch(
                token_indices, WINDOW_SIZE, BATCH_SIZE, neg_sampling_table, NEGATIVE_SAMPLES
            )
            
            batch_loss, grad_U, grad_V_pos, grad_V_neg, center_idx, context_idx, neg_idx = \
                compute_batch_loss_and_gradients(U, V, center_indices, context_indices, negative_indices)
            
            total_loss += batch_loss
            current_step = epoch * num_batches + batch_idx
            lr = get_learning_rate(INITIAL_LR, current_step, total_steps)

            update_embeddings_batch(U, V, center_idx, context_idx, neg_idx,
                                   grad_U, grad_V_pos, grad_V_neg, lr)
            
            
            if batch_idx % 100 == 0:
                avg_loss = total_loss / ((batch_idx + 1) * BATCH_SIZE)
                log_file.write(f"{epoch + 1},{batch_idx},{num_batches},{batch_loss:.6f},{avg_loss:.6f},{lr:.8f}\n")
                log_file.flush()
                print(f"  Progress: {batch_idx}/{num_batches} batches, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
        
        avg_loss = total_loss / (num_batches * BATCH_SIZE)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    log_file.close()
    print(f"Training log saved to {log_filename}")
    
    return U, V, word2idx, idx2word

# Utility functions for finding similar words
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def find_similar_words(word, word2idx, idx2word, embeddings, top_k=5):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary")
        return
    word_idx = word2idx[word]
    word_vec = embeddings[word_idx]
    similarities = []

    for idx in range(len(embeddings)):
        if idx == word_idx:
            continue
        sim = cosine_similarity(word_vec, embeddings[idx])
        similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost similar words to '{word}':")
    for idx, sim in similarities[:top_k]:
        print(f"  {idx2word[idx]}: {sim:.4f}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train Word2Vec embeddings")
    parser.add_argument("--train", action="store_true", help="Train the Word2Vec model")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode to find similar words")
    args = parser.parse_args()
    
    if args.train:
        U, V, word2idx, idx2word = train_word2vec("wiki-data-small.txt")
        
        print("\nSaving embeddings...")
        np.save("artifacts/word_embeddings_U.npy", U)
        np.save("artifacts/word_embeddings_V.npy", V)
        with open("artifacts/vocabulary.txt", "w") as f:
            for idx in range(len(idx2word)):
                f.write(f"{idx2word[idx]}\n")
        print("Training complete!")

        print("\nTesting similar words:")
        test_words = ["king", "queen", "man", "woman", "computer"]
        for word in test_words:
            if word in word2idx:
                find_similar_words(word, word2idx, idx2word, U, top_k=5)

    if args.interactive:
        U = np.load("artifacts/word_embeddings_U.npy")
        with open("artifacts/vocabulary.txt", "r") as f:
            vocab = [line.strip() for line in f]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        print("Enter your own word to find similar words (or 'exit' to quit):")
        while True:
            user_word = input("Word: ").strip()
            if user_word.lower() == "exit":
                break
            if user_word in word2idx:
                find_similar_words(user_word, word2idx, idx2word, U, top_k=5)
            else:
                print(f"Word '{user_word}' not in vocabulary. Try another word.")