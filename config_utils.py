import argparse
import configparser
import random
import numpy as np

# Convenient lookup table for various paths.
# NOTE: Also used by the parser in main.
data_paths = {
    # "trial_answers": 'friends/Answers_for_trial_data/answer.txt',    # No longer used
    "trial": {"episode": 'data/friends/friends.trial.episode_delim.conll',
              "scene": 'data/friends/friends.trial.scene_delim.conll',
              },
    "train": {"episode": 'data/friends/friends.train.episode_delim.conll',
              "scene": 'data/friends/friends.train.scene_delim.conll',
              },
    "test": {"episode": 'data/friends/friends.test.episode_delim.conll',
             "scene": 'data/friends/friends.test.scene_delim.conll',
             },
    "entity_map": 'data/friends/friends_entity_map.txt',
    "embeddings": {"google_news": 'data/GoogleNews-vectors-negative300.bin.gz',
               },
    }


def settings_from_config(file, random_sample=False):
    config = read_config_file(file)

    sampled_params = {}
    fixed_params = {}

    settings = {}
    for section in config.keys():
        settings[section.lower()] = {}
        for param in config[section].keys():
            value = None
            # Either use defaults or sample randomly in the right manner
            if not random_sample or 'sample' not in config[section][param]:
                value = config[section][param]['default']
                fixed_params[param] = value
            else:
                values = config[section][param]['sample']
                if isinstance(values, list):
                    value = random.choice(values)
                elif isinstance(values, tuple):
                    if config[section][param]['type'] == int:
                        value = random.randint(values[0], values[1])
                    else:
                        if len(values) > 2 and values[2] == 'log':
                            minimum = max(values[0], 1e-7)  # Avoid error caused by log 0
                            value = 10 ** random.uniform(np.log10(minimum), np.log10(values[1] + 1e-10))
                        else:
                            value = random.uniform(values[0], values[1])
                    ## Copied from old version, in case reimplementing quantized sampling is desired.
                    # gen_randvalue = lambda interval: \
                    #     random.randint(interval[0], interval[1]) if isinstance(interval[0], int) \
                    #         else decimal.Decimal(random.uniform(interval[0], interval[1] + 1e-10)).quantize(
                    #         decimal.Decimal("%s" % interval[0]))
                sampled_params[param] = value
            settings[section.lower()][param] = value
        # Turn dict into namespace
        settings[section.lower()] = argparse.Namespace(**settings[section.lower()])
    # Turn dict into namespace
    settings = argparse.Namespace(**settings)

    # Shortcuts for data directories:
    if str(settings.data.dataset) in data_paths:
        settings.data.dataset = data_paths[settings.data.dataset][settings.data.level]
    if not 'vocabulary' in vars(settings.data):
        settings.data.vocabulary = settings.data.dataset.replace('.conll', '.vocab')
    if not 'entity_map' in vars(settings.data):
        settings.data.entity_map = data_paths['entity_map']
    if not 'folds_dir' in vars(settings.data):
        settings.data.folds_dir = settings.data.dataset.replace('.conll', '_{0}_fold.pkl'.format(settings.data.folds))
    if str(settings.model.token_emb) in data_paths['embeddings']:
        settings.model.token_emb = data_paths['embeddings'][settings.model.token_emb]
    if str(settings.model.speaker_emb) in data_paths['embeddings']:
        settings.model.speaker_emb = data_paths['embeddings'][settings.model.speaker_emb]

    # TODO @Future: hidden state for Bidirectional runs must be adapted to even number, to be safe and consistent.

    # TODO @Future: Set any irrelevant settings (e.g., attention window, when no attention is used) to None.

    return settings, fixed_params, sampled_params


def read_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    settings = {}
    for section in config.sections():
        settings[section] = {}
        for option in config.options(section):
            value = config.get(section,option)
            partition = value.partition('#')
            main = partition[0].split()
            option = option.replace(" ","_")
            default, type = _value_reader(main[0], typed=True)
            settings[section][option] = {'default': default, 'type': type}
            settings[section][option].update([_value_reader(s) for s in main[1:]])
            settings[section][option]['help'] = partition[2].strip()
    return settings


def write_config_file(args, file_name):
    config = configparser.ConfigParser()
    for section in vars(args).keys():
        config.add_section(section)
        for option, value in vars(vars(args)[section]).items():
            if value == True: value = 'yes'
            elif value == False: value = 'no'
            config.set(section, option.replace('_', ' '), str(value))
    with open(file_name, 'w') as configfile:
        config.write(configfile)


def _value_reader(s, typed=False):
    """
    Converts string to bool if possible, otherwise int, otherwise float, otherwise list/interval, otherwise string
    :param s: a string
    :return: the string's interpretation
    """
    if s == 'yes' or s == 'no':
        s = (s == 'yes')
        return (s, 'bool') if typed else s
    if s.count('.') == 0:       # Is it an int?
        try:
            return (int(s), int) if typed else int(s)
        except ValueError:  # Apparently not...
            pass
    try:            # Is it a float, perhaps?
        return (float(s), float) if typed else float(s)
    except ValueError:
        pass        # Nope. That means it's a string:
    # TODO @Future: the following would be safer with regular expression matching
    if '-' in s and len(s.split('-'))==2:    # A linear interval?
        s = [_value_reader(v, False) for v in s.split('-')]
        return ('sample', (s[0], s[1], 'lin'))
    elif '~' in s and len(s.split('~'))==2:  # A log interval?
        s = [_value_reader(v, False) for v in s.split('~')]
        return ('sample', (s[0], s[1], 'log'))
    elif '|' in s:  # A set of options?
        s = [_value_reader(v, False) for v in s.split('|')]
        return ('sample', s)

    return (s, str) if typed else s       # Ordinary string then


def fixed_params_to_string(file):
    _, fixed_params, _ = settings_from_config(file, random_sample=True)
    stringed_params = []
    for key in sorted(fixed_params):
        if not (key == 'phase' or key == 'stop_criterion' or key == 'test_every' or key == 'layers_lstm'):
            if isinstance(fixed_params[key],bool):
                if fixed_params[key]:
                    stringed_params.append(key[:3])
            elif key == 'level' or key == 'optimizer':
                stringed_params.append(fixed_params[key][:3])
            else:
                stringed_params.append(key[:3] + str(fixed_params[key])[:3])
    return '-'.join(stringed_params)
