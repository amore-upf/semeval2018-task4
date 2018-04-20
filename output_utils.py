import re
import torch
import os
import config_utils

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    WHISPER = (3, bcolors.OKBLUE)
    SAY = (2, bcolors.ENDC)
    SHOUT = (1, bcolors.HEADER)

    def __init__(self, args):
        self.run_name = args.run_name
        self.no_files = args.no_files
        self.verbose = args.verbosity

        # Directory structure
        if args.model_dir is not None:
            self.model_dir = os.path.join(args.model_dir, args.subdir)
            self.log_dir = os.path.join(self.model_dir, 'logs')
            self.answer_dir = os.path.join(self.model_dir, 'answers')

        if not self.no_files:
            if args.phase == 'train':
                os.makedirs(self.model_dir, exist_ok=True)
                os.makedirs(self.log_dir, exist_ok=True)
            elif args.phase == 'deploy':
                os.makedirs(self.answer_dir, exist_ok=True)

        # Bookkeeping during training, to be overwritten:
        self.fold_idx = -1

    def save_model(self, model, suffix=""):
        if not self.no_files:
            infix = "--fold" + str(self.fold_idx) if self.fold_idx >= 0 else ""
            filename = self.run_name + infix + suffix + '.pt'
            model_file = os.path.join(self.model_dir, filename)
            torch.save(model.state_dict(), model_file)
            self.whisper("Model saved as "+model_file)

    def save_config(self, settings, config_dir=None):
        if not self.no_files:
            if config_dir is None:
                config_file = os.path.join(self.model_dir, self.run_name + '.ini')
                config_file2 = os.path.join(self.log_dir, self.run_name + '.ini')
                config_utils.write_config_file(settings, config_file)
                config_utils.write_config_file(settings, config_file2)
                self.whisper("Config written to " + config_file)
                self.whisper("Config written to " + config_file2)
            else:
                config_file = os.path.join(config_dir, self.run_name + '.ini')
                config_utils.write_config_file(settings, config_file)
                self.whisper("Config written to " + config_file)

    def say(self, message, level = SAY):
        if self.verbose >= level[0]:
            print(level[1]+str(message)+bcolors.ENDC)

    def whisper(self, message):
        self.say(message, level=Logger.WHISPER)

    def shout(self, message):
        self.say(message, level=Logger.SHOUT)

    def log(self, message):
        if isinstance(message, dict):
            message['fold'] = self.fold_idx
            message['model'] = self.run_name  # TODO @Future: This is a bit redundant, and not currently used.
            result_format_str = '{0[epoch]:5}\t{0[iteration]:5}\t{0[fold]:3d}\t{0[training][loss]:10.7f}\t{0[training][accuracy]:10.4f}\t{0[training][macro_f1_score]:10.4f}\t{0[training][macro_f1_score_main]:10.4f}\t{0[training][total]:7d}\t{0[validation][loss]:10.7f}\t{0[validation][accuracy]:10.4f}\t{0[validation][macro_f1_score]:10.4f}\t{0[validation][macro_f1_score_main]:10.4f}\t{0[validation][total]:7d}\t{0[model]:30}'
            message = result_format_str.format(message)
        if not self.no_files:
            filename = self.run_name+'.log'
            log_file = os.path.join(self.log_dir, filename)
            with open(log_file, "a", encoding="utf-8") as outf:
                print(message, file=outf)
            outf.close()
            self.whisper("Logged: "+message)
        else:
            self.shout(message)

    def write_answers_csv(self, data_path, predictions, model_suffix="", config=None):
        if not self.no_files:
            output_dir = re.sub("_delim.conll", "", os.path.basename(data_path)).replace(".", "_")
            output_dir = os.path.join(self.answer_dir, output_dir)
            os.makedirs(output_dir, exist_ok=True)

            pred_fname = os.path.join(output_dir, '{0}.csv'.format(self.run_name + model_suffix))
            predictions_file = open(pred_fname, 'w')

            header = ['index', 'prediction', 'target'][0:len(list(predictions)[0])]
            predictions_file.write('# '+data_path+'\n')        # This line is read during evaluate phase.
            predictions_file.write('\t'.join(header)+'\n')
            for p in predictions:
                predictions_file.write('\t'.join([str(x) for x in p])+'\n')

            predictions_file.close()

            self.whisper("Predictions written into {0}.".format(pred_fname))

            if config is not None:  # Optionally save a copy of the config:
                self.save_config(config, config_dir=os.path.dirname(pred_fname))

            return pred_fname
