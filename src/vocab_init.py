# --- imports --- #

import numpy as np
import spacy


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
