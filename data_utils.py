import operator
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import collections


# TODO @Future: Vocab class with methods get_idx(word) / get_word(idx); same for Entity; make constants fields of classes
UNKNOWN_WORD_IDX = 0
UNKNOWN = "unknown_word"
MAIN_ENTITIES = [59, 183, 335, 306, 248, 292]
NON_ENTITY_MENTION = -1
ENTITY_MENTION = 999 # TODO @Future: change (but maybe better to not use negative idx for now, in case expressions like idx < 0)
DUMMY_ENTITY_IDX = -1
OTHER_ENTITY = 401 # TODO @Future: ideally set this automatically depending on entity keys; and check if there's an 'other' category already.


def get_embeddings_path_for_vocab(embeddings_fname, vocab_fname):
    return re.sub("\.(txt|vocab)$", "", vocab_fname)+"__"+os.path.basename(embeddings_fname).split(".")[0] + ".npy"


def load_entity_map(filepath):
    """
    :param filepath: string path to file containing the entity map as provided by semeval task organizers.
    :return: pair of back-and-forth mappings (entity_idx_to_name, entity_name_to_idx)
    The latter contains additional, first-name-only entries for the main characters.
    """
    entity_idx_to_name = pd.read_csv(filepath,
                                    sep='\t',
                                    usecols=[1],
                                    squeeze=True,
                                    header=None,
                                    names=['name']
                                    )

    entity_name_to_idx = {name: idx for idx, name in entity_idx_to_name.items()}
    # Manually add first-name keys for the main characters (for these are used sometimes in the data files)
    entity_name_to_idx['Chandler'] = entity_name_to_idx['Chandler Bing']
    entity_name_to_idx['Phoebe'] = entity_name_to_idx['Phoebe Buffay']
    entity_name_to_idx['Joey'] = entity_name_to_idx['Joey Tribbiani']
    entity_name_to_idx['Ross'] = entity_name_to_idx['Ross Geller']
    entity_name_to_idx['Rachel'] = entity_name_to_idx['Rachel Green']
    entity_name_to_idx['Monica'] = entity_name_to_idx['Monica Geller']

    return entity_idx_to_name, entity_name_to_idx

def load_data(filepath, entity_name_to_idx, with_ling_info=False, with_keys=None, logger=None):
    """
    :param filepath: path of file containing the language data the model is trained/tested on
    :param entity_name_to_idx: a dictionary mapping names to indices
    :param with_lemma: True if storing the lemma of the token (default: False)
    :return: a pandas dataframe [episode (string), scene (int), position (int), token (string), speaker (int list), entity (int)]
    Notes:
    * Speaker column contains lists of integers because sometimes (very rarely) multiple speakers are specified.
    * -1 in speaker column means unknown speaker; -1 in entity column means no entity was mentioned.
    * There are some multi-word mentions (VERY rare), in which case all entity IDs are ignored except for the last word.
    * First tried NLTK's CONLL corpus reader, but this felt a bit clunky (and needs some preprocessing to ignore '#'-lines).
        # from nltk.corpus.reader import conll
        # reader = conll.ConllCorpusReader('friends_data/Trial_and_Training_data_and_Entity_mapping',
                                           'friends.trial.episode_delim.conll')
    """
    # Make (the number of) return values dependent on whether parameter `with_keys` was specified, so as to ensure compatibility with existing code (which expects `data` being returned only).
    if with_keys is None:
        data, data_has_keys = _load_data(filepath, entity_name_to_idx, with_keys, with_ling_info, logger)
        return data, data_has_keys
    else:
        data, data_has_keys = _load_data(filepath, entity_name_to_idx, with_keys, with_ling_info, logger)
        if with_keys == True:
            if data_has_keys == False and logger is not None:
                logger.shout('Warning: Entity keys were requested but were not found in data.')
        return data
            
                
def _load_data(filepath, entity_name_to_idx, with_keys, with_ling_info=False, logger=None):
    """
    :param filepath: path of file containing the language data the model is trained/tested on
    :param entity_name_to_idx: a dictionary mapping names to indices
    :param with_lemma: True if storing the lemma of the token (default: False)
    :return: a pandas dataframe [episode (string), scene (int), position (int), token (string), speaker (int list), entity (int)]
    Notes:
    * Speaker column contains lists of integers because sometimes (very rarely) multiple speakers are specified.
    * -1 in speaker column means unknown speaker; -1 in entity column means no entity was mentioned.
    * There are some multi-word mentions (VERY rare), in which case all entity IDs are ignored except for the last word.
    * First tried NLTK's CONLL corpus reader, but this felt a bit clunky (and needs some preprocessing to ignore '#'-lines).
        # from nltk.corpus.reader import conll
        # reader = conll.ConllCorpusReader('friends_data/Trial_and_Training_data_and_Entity_mapping',
                                           'friends.trial.episode_delim.conll')
    """        
    cols = [0, 1, 2, 3, 9, 11] if not with_ling_info else [0, 1, 2, 3, 4, 6, 9, 11]
    if not with_ling_info:
        names_cols = ['episode', 'scene', 'position', 'token', 'speaker', 'entity']
    else:
        names_cols = ['episode', 'scene', 'position', 'token', 'pos', 'lemma','speaker', 'entity']

    # Loads data more general - without assumption of an existing entity map
    if entity_name_to_idx is not None:
        # Data will be read using converters for speaker and entity column:
        # Speaker_converter replaces lists of speaker names by list of indices, -1 for unknown:
        speaker_converter = lambda x: [int(entity_name_to_idx.get(name, entity_name_to_idx["Unknown"]))
                                                           for name in x.replace('_', ' ').split(', ')]
    else:
        speaker_converter = lambda x: [y for y in x]

    # Entity_converter can either (try to) read entity keys or merely mention markers:
    # if not '.nokey' in filepath:  # <-- this seemed a bit risky; also we may want to be able to test on keyed data.
    if with_keys is not False:
        # replaces strings like '(384)' and '-' by integers 384 and NON_ENTITY_MENTION, respectively.
        entity_converter = lambda x: int(re.sub("[^0-9]", "", x) if re.match("\(*\d+\)",x) else NON_ENTITY_MENTION)
        if '.nokey' in filepath:
            if logger is not None and with_keys is True:
                logger.shout('Warning: Entity keys were requested but data path contains \'.nokey\'.')
            with_keys = False
    if with_keys is False:
        entity_converter = lambda x: ENTITY_MENTION if x == '(-1)' or x == '-1)' or re.match("\(*\d+\)",x) else NON_ENTITY_MENTION

    data = pd.read_csv(filepath,
                       delim_whitespace = True,
                       comment='#',
                       usecols=cols,
                       names=names_cols,
                       converters = {'speaker': speaker_converter, 'entity': entity_converter},
                       quoting = 3,  # no special treatment for quotation marks (error otherwise)
                       )
    entity_keys_unique = data["entity"].unique()
    if list(entity_keys_unique) == [-1]:
        # data without keys, i.e. with -1 indicating entity mentions, provided.
        data[["entity"]] = ENTITY_MENTION
        with_keys = False
    else:
        with_keys = (ENTITY_MENTION not in entity_keys_unique) or (len(entity_keys_unique) > 2)

    if logger is not None:
        logger.whisper('Data loaded from '+filepath+' ('+str(len(data))+' lines)')

    return data, with_keys


def load_answers(filepath):
    """
    :param filepath: path of file containing reference data
    :return: a pandas Series object containing ints
    """
    answers = pd.read_csv(filepath, names=['entity'], squeeze=True)
    return answers


def extract_answers(data):
    """
    Extracts an answer list (like the one in answer.txt) from the data.
    :param data: a Pandas dataframe containing an 'entity' column.
    :return: a Pandas Series of integers extracted from the 'entity' column.
    """
    answers = data[['entity']].squeeze()
    return answers[answers != NON_ENTITY_MENTION].reset_index(drop=True)


def extract_vocabulary(data, entity_map_path=None, logger=None):
    """
    Collect all tokens from data in indexed vocabulary.
    :param data_path: path of file containing the language data the model is trained/tested on, and from which the vocabulary is extracted.
    :param entity_map_path: The name of the file containing the relevant vocabulary (entity names). 
            Each line contains the word idx and the word, separated by tabs ("\t").
    :return: token-to-id mapping (np.array), id-to-token mapping (dictionary)
    """
    vocabulary_idx_to_word = sorted(data['token'].str.lower().drop_duplicates().values)
    vocabulary_idx_to_word.insert(UNKNOWN_WORD_IDX, UNKNOWN)
    vocabulary_word_to_idx = {word: idx for idx, word in enumerate(vocabulary_idx_to_word)}
    return vocabulary_idx_to_word, vocabulary_word_to_idx


def load_vocabulary(filepath, sep="\t", logger=None):
    if not os.path.exists(filepath):
        if logger is not None:
            logger.shout("Vocabulary cannot be loaded from {0}. File does not exist.\n".format(filepath))
        return None, None
    vocabulary_word_to_idx = {line.strip().split(sep)[1]:int(line.split(sep)[0]) for line in open(filepath)}
    # just in case the vocab file does not contain the words in order of their ids:
    
    # [bug fix from 09/04/2018: insert "unknown_word" at position 0] 
    # TODO @Future: Check whether this bug had an influence on models before the fix (the two data structures are only used in create_input_vectors(), and token_ids (from vocab_word_to_idx, see create_input_vectors) don't seem to be used for accessing vocab_idx_to_word at any point. BUT vocab_idx_to_word is used for storing the vocab ...
    #vocabulary_idx_to_word = sorted(vocabulary_word_to_idx)
    vocabulary_idx_to_word = [word_idx[0] for word_idx in sorted(vocabulary_word_to_idx.items(), key=operator.itemgetter(1))]

    return vocabulary_idx_to_word, vocabulary_word_to_idx


def save_vocabulary(vocabulary_idx_to_word, outfname):
    with open(outfname, "w") as fout:
        for (idx, word) in enumerate(vocabulary_idx_to_word):
            fout.write("{0}\t{1}\n".format(idx, word))


def get_vocabulary(vocab_fname, logger=None, extract_from=None, entity_map_path=None):
    if os.path.exists(vocab_fname):
        vocabulary_idx_to_word, vocabulary_word_to_idx = load_vocabulary(vocab_fname, logger=logger)
        if logger is not None:
            logger.whisper('Vocabulary loaded from '+vocab_fname)
    elif extract_from is None:
        if logger is not None:
            logger.shout('ERROR: Vocabulary (' + vocab_fname + ') does not exist, and no data was provided to extract it from.')
    else:        
        vocabulary_idx_to_word, vocabulary_word_to_idx = extract_vocabulary(extract_from, entity_map_path, logger)
        save_vocabulary(vocabulary_idx_to_word, vocab_fname)
        if logger is not None:
            logger.whisper('Vocabulary extracted from data; saved to ' + vocab_fname)
    return vocabulary_idx_to_word, vocabulary_word_to_idx


def get_sequence_bounds(data, level):
    """
    Split data into sequences of type level (scenes or episodes)
    :param data: data in DataFrame with columns 'scene' and 'episode'
    :return: array of pairs [start, end] such that data.loc[start,end] are the sequences (scenes or episodes).
    """
    sequences = []
    start_current_sequence = 0
    new_sequence = True
    data_length = len(data)
    last_row = data_length - 1
    for row_number in range(data_length):
        # Get token info
        current_episode, current_scene = data.loc[row_number][:2]
        if new_sequence:
            start_current_sequence = row_number
            new_sequence = False
        if row_number != last_row:
            next_row = data.loc[row_number + 1]
            #Check if end of scene (end of episode is also end of scene)
            if current_episode != next_row.episode:
                new_sequence = True
            elif level == 'scene' and current_scene != next_row.scene:
                new_sequence = True
        else:
            #Last token in data
            new_sequence = True
        if new_sequence:
            # store start and end point in the sequences list
            sequences.append([start_current_sequence, row_number + 1])
    return np.array(sequences)


def _create_folds_of_sequences(data, sequence_bounds, num_folds):
    # Results:
    # trial_scene:   fold_weights = [117 117  81  60  44  50  44  43  40  42]
    # trial_episode: fold_weights = [423 212   0   0   0   0   0   0   0   0]
    # train_scene:   fold_weights = [1334 1338 1336 1340 1337 1334 1334 1333 1333 1333]
    # train_episode: fold_weights = [1378 1368 1367 1399 1372 1399 1210 1373 1210 1212]

    # If the file didn't exist yet, create folds
    mention_counts = np.empty(len(sequence_bounds), dtype=int)
    total_mentions = 0
    for i, seq in enumerate(sequence_bounds):
        mention_count = np.sum(data['entity'].loc[seq[0]:seq[1]] != NON_ENTITY_MENTION)
        mention_counts[i] = mention_count
        total_mentions += mention_count

    weights = mention_counts    # Weights could be defined in a smarter way

    # Greedy approach: add sequences, from heavy to light, to currently lightest fold
    sorted_ids = weights.argsort()[::-1]
    folds_ids = [[] for i in range(num_folds)]
    fold_weights = np.zeros(num_folds, dtype=int)
    for idx in sorted_ids:
        lightest_fold_idx = np.argmin(fold_weights)
        folds_ids[lightest_fold_idx].append(idx)
        fold_weights[lightest_fold_idx] += mention_counts[idx]

    # Retrieve set of sequence bounds for each fold
    folds = []
    for fold_idx in folds_ids:
        folds.append(sequence_bounds[np.array(fold_idx, dtype=int)])

    return folds, fold_weights


def get_cv_folds(data, settings, logger=None):
    # Load or create folds for cross-validation for the current dataset
    if os.path.isfile(settings.folds_dir):
        with open(settings.folds_dir, 'rb') as f:
            folds = pickle.load(f)
        if logger is not None:
            logger.whisper('Folds loaded from ' + settings.folds_dir)
    else:
        if logger is not None:
            logger.whisper('Creating folds for cross-validation...')
        sequence_bounds = get_sequence_bounds(data, settings.level)
        folds, fold_weights = _create_folds_of_sequences(data, sequence_bounds, settings.folds)
        with open(settings.folds_dir, 'wb') as f:
            pickle.dump(folds, f)
        if logger is not None:
            logger.whisper('  ...Folds saved to ' + settings.folds_dir + ' (weights: ' + str(fold_weights)+')')
    return folds


def create_input_vectors(data, vocabulary_id_to_word, vocabulary_word_to_id):
    """
    Create token vocabulary and convert data to numpy array with word (token or entity/speaker) token_ids.
    """
    # TODO @Future Append info about context, say, other entities present in the scene (requires adaptation of model, too).
    train_data_length = len(data.index)
    token_ids = []

    num_words = np.zeros(train_data_length, dtype=int)
    for token_index in range(train_data_length):
        token_id = vocabulary_word_to_id.get(data['token'][token_index].lower(), UNKNOWN_WORD_IDX)
        token_speaker_ids = [token_id]
        if token_id >= len(vocabulary_id_to_word):
            sys.stderr.write("Token idx >= size(vocab)")
            quit()
        speaker_ids = data['speaker'][token_index]
        num_words[token_index] = len(speaker_ids) + 1  # plus 1 token at index 0
        token_speaker_ids.extend(speaker_ids)
        # TODO @Future: this makes +/- speaker switch (if wished) inflexible
        token_ids.append(token_speaker_ids)
        # this is redundant right now, since all tokens are collected, but if only entity mentions were collected this list would be a sublist of all token indices

    max_num_speaker = num_words.max()
    mask = np.arange(max_num_speaker) < num_words[:, None]
    input_vecs = np.zeros(mask.shape, dtype=int) + DUMMY_ENTITY_IDX
    input_vecs[mask] = np.concatenate(np.array(token_ids))
    targets = data['entity'].values.astype(int)

    return input_vecs, targets


def transform_labels_into_names(labels, entity_idx_to_name):
    """
    Trasform a list of number labels to the related names of characters
    :param labels:
    :param entity_idx_to_name:
    :return: list of names labels
    """
    names_labels = []
    for label in labels:
        if label < len(entity_idx_to_name):
            names_labels.append(entity_idx_to_name[label].replace(' ','_'))
        elif label == OTHER_ENTITY:
            names_labels.append('Other')
    return names_labels


def mask_ids_by_token_type(data, indices):
    """
    Filter indices by token type, by default
    :param data: dialogue data
    :return: lists of mentions number that are proper or common nouns respectively
    """
    token_types = ['all','common noun', 'proper noun', 'pronoun', 'pronoun 1pers', 'pronoun 2pers', 'pronoun 3pers']
    mask = {token_type: np.zeros(len(indices), dtype=np.bool) for token_type in token_types}

    for j,i in enumerate(indices):
        pos = data.loc[i]['pos']
        mask['common noun'][j] = (pos == 'NN')
        mask['proper noun'][j] = (pos == 'NNP')
        m = data.loc[i]['token'].lower()
        mask['pronoun 1pers'][j] = (m in ['i', 'me', 'my', 'myself', 'mine'])
        mask['pronoun 2pers'][j] = (m in ['you', 'your', 'yourself', 'yours'])
        mask['pronoun 3pers'][j] = (m in ['she', 'her','herself', 'hers', 'he', 'him', 'himself', 'his', 'it', 'itself', 'its'])
        mask['pronoun'][j] = mask['pronoun 1pers'][j] or mask['pronoun 2pers'][j] or mask['pronoun 3pers'][j]
        mask['all'][j] = True

    return mask


def get_counts_entities(labels):
    """
    Get counts of mentions for each entity
    :param labels: gold labels
    :return: counts for each entity
    """
    counter_entities = collections.Counter(labels)
    counts_main_entities = sum([counter_entities[e] for e in MAIN_ENTITIES])
    main_entities_portion = 100 * counts_main_entities / len(labels)
    return counter_entities, main_entities_portion


def get_labels(data):
    """
    Get gold labels in data
    :param data: dialogue data
    :return: lists of labels
    """
    labels = []
    for i in range(len(data)):
        e = data.loc[i]['entity']
        if e != NON_ENTITY_MENTION:
            labels.append(e)
    return labels


def get_mentions(data):
    """
    Get mentions  in data
    :param data: dialogue data
    :return: list of mentions
    """
    mentions = []
    for i in range(len(data)):
        if data.loc[i]['entity'] != NON_ENTITY_MENTION:
            mentions.append(data.loc[i]['token'].lower())
    return mentions


def read_answers_csv(path, logger=None):
    answers = pd.read_csv(path, delim_whitespace=True, comment='#', dtype={'index': int, 'prediction': int, 'target': int})
    indices = answers['index']
    prediction = answers['prediction']
    target = answers['target']
    if logger is not None:
        logger.whisper('Predictions (and targets and indices) read from '+path)
    return indices, prediction, target


def data_summary(data):
    summary = "DATA SUMMARY:\n"
    mentions = get_mentions(data)
    counts_mentions_type = collections.Counter(mentions)
    targets = get_labels(data)
    summary += 'most common referring expressions: {0}\n'.format(counts_mentions_type.most_common(20))
    counts_entities, percentage_main_entities = get_counts_entities(targets)
    summary += 'most common entities: {0}\n'.format(counts_entities.most_common(20))
    summary += 'mentions of main entities: {0}%\n'.format(percentage_main_entities)
    return summary