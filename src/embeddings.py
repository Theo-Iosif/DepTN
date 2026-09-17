# --- imports --- #

import numpy as np
import spacy
from sklearn.decomposition import PCA

from helper_functions import generate_haar_isometric

## ---------------------------------------------------- ##
## ===== VOCABULARY AND DEPENDENCY INITIALIZATION ===== ##

# --- load english module --- #

nlp = spacy.load("en_core_web_sm")

    
# --- vocabulary and dependency initiation --- #

def load_vocabulary():
    ''' 
    Loads the vocabulary from the spaCy model.

    output: (list, int): A tuple containing the vocabulary list and its length.
    '''

    vocab = [string for string in nlp.vocab.strings]
    
    return vocab, len(vocab)

def load_dependencies():
    '''
    Loads the dependency labels from the spaCy model.

    output: (list, int): A tuple containing the dependency labels and their length.
    '''

    dep = nlp.get_pipe("parser").labels

    return dep, len(dep)

def create_dictionary(vocab, dep):
    '''
    Creates a dictionary mapping token IDs to their strings and dependency labels.

    input: vocab (list): The vocabulary list.
           dep (list): The dependency labels list.

    output: dict: A dictionary mapping token IDs to (string, dep_label) tuples.
    '''

    vocab_dict = {word: i for i, word in enumerate(vocab)}
    dep_dict = {label: i for i, label in enumerate(dep)}

    return vocab_dict, dep_dict


## ---------------------------------------------------- ##
## =====   VOCABULARY AND DEPENDENCY EMBEDDINGS   ===== ##

# --- vocabulary embeddings --- #

#   -> random embedding

def vocab_embedding_random(target_vocab, input_dim, normalize=True):
    '''
    Generate random embeddings for vocabulary words. Assumes the embeddings match the vocabulary dictionary assignment.

    input: target_vocab (set): The set of words in the vocabulary.
           input_dim (int): The dimensionality of the word embeddings.
           normalize (bool): Whether to normalize the embeddings to unit length.

    output: (dict): Dictionary mapping words to their embeddings.
    '''

    embedding_dict = {word: np.random.rand(input_dim) for word in target_vocab}

    if normalize:
        for word, vector in embedding_dict.items():
            embedding_dict[word] = vector / np.linalg.norm(vector)

    return embedding_dict

#   -> pre-trained embedding based on local file

def vocab_embedding_pretrained_fromlocal(embedding_file, target_vocab, normalize=True):
    '''
    Load pre-trained embeddings from a local file and filter them based on the target vocabulary.

    input: embedding_file: path to the pre-trained embedding file
           target_vocab: set of strings containing words in your corpus
           normalize: boolean indicating whether to normalize the embeddings

    output: (dict): dictionary mapping words to their embeddings
    '''
    embedding_dict = {}
    
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            if word in target_vocab:
                vector = np.asarray(parts[1:], dtype='float32')
                if normalize:
                    vector = vector / np.linalg.norm(vector)
                embedding_dict[word] = vector
                
    return embedding_dict # dictionary with mutable values for embeddings

# reduce dimensionality of embeddings using PCA

def reduce_embedding_dimensions(embedding_dict, input_dim):
    '''
    Reduces the dimensionality of a word embedding dictionary using PCA.

    input: embedding_dict (dict): A dictionary mapping words to their embeddings.
           input_dim (int): The target dimensionality for the embeddings.

    output: (dict): A dictionary mapping words to their reduced embeddings.
    '''

    # check if the input_dim is less than the current dimension of the embeddings
    current_dim = len(next(iter(embedding_dict.values()))) # % check bug
    if input_dim >= current_dim:
        raise ValueError(f"Input dimension {input_dim} must be less than the current embedding dimension {current_dim}.")
    
    # 1. Extract words and vectors into aligned arrays
    words = list(embedding_dict.keys())
    vectors = np.array(list(embedding_dict.values()))
    
    # 2. Apply PCA
    pca = PCA(n_components=input_dim)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Optional but recommended: Check how much variance is retained
    variance_retained = np.sum(pca.explained_variance_ratio_)
    print(f"Variance retained with {input_dim} dimensions: {variance_retained:.2%}")
    
    # 3. Rebuild the dictionary mapping
    reduced_dict = {word: reduced_vectors[i] for i, word in enumerate(words)}
    
    return reduced_dict


# --- dependency embeddings --- #

def dep_embedding_random(target_dep, inner_dim, isometry = True):
    '''
    Generate random embeddings for dependency labels.

    input: target_dep (list): The list of dependency labels.
           inner_dim (int): The dimensionality of the word embeddings.
           isometry (bool): Whether to generate Haar random matrices instead of Euclidean random matrices.

    output: (dict): Dictionary mapping dependency labels to their embeddings.
    '''
    if isometry:
        dep_embedding_dict = {label: generate_haar_isometric(inner_dim, inner_dim) for label in target_dep}
    else:
        dep_embedding_dict = {label: np.random.rand(inner_dim, inner_dim) for label in target_dep}

    return dep_embedding_dict