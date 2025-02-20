from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import random

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    sorted_vocab: List[int] = sorted(word_counts, key = lambda x: word_counts[x], reverse=True)
    
    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {sorted_vocab.index(word): word for word in sorted_vocab}        
    vocab_to_int: Dict[str, int] = {word: sorted_vocab.index(word) for word in sorted_vocab}

    return vocab_to_int, int_to_vocab

def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words.
    
    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

        
    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # TODO
    # Convert words to integers
    if not words:
        return [], {}
    
    counts = dict(Counter(words))
    total = len(words)
        
    freqs: Dict[str, float] = {word: word_count/total for word, word_count in counts.items()}

    subsampling_prob = {word: (1-torch.sqrt(torch.tensor(threshold/word_freq))) for word, word_freq in freqs.items()}
    
    train_words: List[int] = []
    for word in words:
            if torch.rand(1).item() > subsampling_prob[word]:
                train_words.append(vocab_to_int[word])

    return train_words, freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    r = random.randint(1, window_size)
    start_idx = max(0, idx - r)
    end_idx = min(len(words), idx + r + 1)
    
    return [words[i] for i in range(start_idx, end_idx) if i != idx]

def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Generator[None, None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """

    # TODO
    for idx in range(0, len(words), batch_size):
        inputs: List[int] = []
        targets: List[int] = []
        target_words = get_target(words, idx, window_size)
        for w in target_words:
            inputs.append(words[idx])
            targets.append(w)

        yield (inputs, targets)

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO
    valid_examples: torch.Tensor = []
    i = 0
    while i < valid_size:
        j = random.randint(0,valid_window) 
        if j not in valid_examples:
            valid_examples.append(j)
            i+=1

    valid_examples = torch.tensor(valid_examples)

    valid_vectors = embedding(valid_examples)

    matrix_embeddings = embedding.weight 
    matrix_norm = matrix_embeddings/torch.norm(matrix_embeddings, dim=1, keepdim=True)
    
    valid_vectors_norm = valid_vectors/torch.norm(valid_vectors, dim=1, keepdim=True)
    
    similarities = torch.matmul(valid_vectors_norm, matrix_norm.T)  

    return valid_examples, similarities