import os
import sys

import numpy as np

import gensim
from gensim.models import word2vec

import data_utils
from config_utils import data_paths

DECREASE_FACTOR=1e-4 # TODO @Future: there should be a smarter way

def load_word2vec_embeddings(filename):
    binary_file = ".bin" in filename
    return gensim.models.KeyedVectors.load_word2vec_format(filename, binary=binary_file)
    
    
def filter_embeddings(word_vectors, vocabulary_idx_to_word, normalise_word=False, rnd_init=False):
    """
    :param normalise_word: set words to lowercase and replace whitespace by '_'
    :param rnd_init: If True, set initial weights to random numbers drawn from uniform distribution, otherwise set weights to zero.
    """
    unknown_inds = []
    found_inds = []
    if rnd_init:
        rel_vectors = np.random.rand(len(vocabulary_idx_to_word), word_vectors.vector_size)*DECREASE_FACTOR
    else:
        rel_vectors = np.zeros(shape=(len(vocabulary_idx_to_word), word_vectors.vector_size))
    for (idx, word) in enumerate(vocabulary_idx_to_word):
        if normalise_word:
            word = word.replace(" ", "_").lower()
        if word in word_vectors:
            rel_vectors[idx,:] = word_vectors.wv[word]
            found_inds.append(idx)
        else:
            unknown_inds.append(idx)
    return rel_vectors, unknown_inds, found_inds

def fill_missing_embeddings(word_embeddings, unk_inds, found_inds):
    """
    For unknown entities: add average emb vector of found entities to their random initialisation
    TODO @Future: Is it better to initialize these as zeros instead of averages?
    """
    avg_entity_vecs = np.mean(word_embeddings[found_inds],0)
    word_embeddings[unk_inds] += avg_entity_vecs*1e-2
    

def load_word_embeddings(embeddings_fname, training_datapath, training_data, logger=None):
    """
    :param embeddings_fname: The name of the file containing pre-trained embeddings. 
            E.g., the Google-news w2v embeddings
    :param training_datapath: The name of the file containing the training data for 
            a model which uses word embeddings (loaded from embeddings_fname). 
    """
    # vocab_fname: The name of the file containing the relevant vocabulary. 
    #        Each line contains the word idx and the word, separated by tabs ("\t"). 
    vocab_fname = training_datapath.replace(".conll", ".vocab")
    word_emb_fname = data_utils.get_embeddings_path_for_vocab(embeddings_fname, vocab_fname)
    if os.path.exists(word_emb_fname):
        if logger:
            logger.whisper("Loading token embedding from {0}".format(word_emb_fname))
        word_embeddings = np.load(word_emb_fname)
    else:
        vocabulary_idx_to_word,_ = data_utils.get_vocabulary(vocab_fname, extract_from=training_data, logger=logger)
        all_word_vectors = load_word2vec_embeddings(embeddings_fname)
        word_embeddings,_,_ = filter_embeddings(all_word_vectors, vocabulary_idx_to_word)
        save_word_embeddings(word_embeddings, word_emb_fname)
    return word_embeddings


def load_entity_embeddings(embeddings_fname, vocab_fname, logger=None):
    """
    :param embeddings_fname: The name of the file containing pre-trained embeddings. 
            E.g., the Google-news w2v embeddings
    :param vocab_fname: The name of the file containing the relevant vocabulary (entity names). 
            Each line contains the word idx and the word, separated by tabs ("\t").
    """
    if not embeddings_fname.endswith(".npy"):
        embeddings_fname = data_utils.get_embeddings_path_for_vocab(embeddings_fname, vocab_fname)

    if os.path.exists(embeddings_fname):
        if not logger is None:
            logger.whisper("Loading entity embedding from {0}".format(embeddings_fname))
        word_embeddings = np.load(embeddings_fname)
    """
    # The model does not use embeddings (yet) which were extracted from some other source
    else:
        vocabulary_idx_to_word,_ = data_utils.load_vocabulary(vocab_fname)
        all_entity_vectors = load_word2vec_embeddings(embeddings_fname)
        word_embeddings, unk_inds, found_inds = filter_embeddings(all_entity_vectors, vocabulary_idx_to_word, normalise_word=True, rnd_init=True)
        fill_missing_embeddings(word_embeddings, unk_inds, found_inds)
        save_word_embeddings(word_embeddings, embeddings_fname)
    """
    return word_embeddings


def save_word_embeddings(word_embeddings, outfname, logger=None):
    np.save(outfname, word_embeddings)
    if not logger is None:
        logger.whisper("Embeddings saved in \n\t{0}".format(outfname))   
