from random import shuffle

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.tensor
import tensor_utils

from data_utils import NON_ENTITY_MENTION, MAIN_ENTITIES, OTHER_ENTITY
from sklearn.metrics import f1_score, accuracy_score
import warnings



def yield_padded_batch_and_bounds(X, y, seq_bounds, batch_size, chunk_size, shuffle_data, use_cuda=False):
    """
    Split list of scenes/episodes in batches
    :param data: list of episodes or scenes in DataFrame format
    :param batch_size: batch size
    param chunk_size: chunk size, set chunk_size==-1 to set it to max_seq_length (for testing)
    :param shuffle: if True shuffles the data before splitting it into batches
    :return: list of batches for training
    """
    # TODO @Future: Maybe random offsets for starting points of each scene/episode.
    seq_ids = list(range(len(seq_bounds)))
    if shuffle_data:
        shuffle(seq_ids)
    seq_ids_per_batch = [seq_ids[i:i + batch_size] for i in range(0, len(seq_ids), batch_size)]
    seq_bounds_per_batch = [seq_bounds[batch_ids] for batch_ids in seq_ids_per_batch]
    if shuffle_data:    # TODO @Future: ultimately necessary when we use pack_padded, in which case it must be unsorted afterwards.
        seq_bounds_per_batch = [batch[np.argsort([seq[1] - seq[0] for seq in batch])[::-1]] for batch in
                                seq_bounds_per_batch]
    for i, b in enumerate(seq_bounds_per_batch):
        seq_lengths = b[:, 1] - b[:, 0]

        max_seq_length = max(seq_lengths).item()
        # NOTE: item() removes numpy wrapping, which would be harmful later

        # TODO @Future: Maybe it's more efficient to already move the data to CUDA earlier.
        # Obtain all sequences, put them into a padded tensor
        X_sequences = [torch.from_numpy(X[seq[0]:seq[1]]) for seq in b]
        X_padded_sequences = [torch.cat([t, -1*(torch.ones(max_seq_length - len(t), X.shape[1]).long())]) for t in X_sequences]
        X_padded_batch = torch.stack(X_padded_sequences)

        if use_cuda:
            X_padded_batch = X_padded_batch.cuda()

        # This is maintained ONLY for compatibility with TESTING mode, but even there it could be avoided.
        if chunk_size == -1:  # sequences are split into chunks only during training
            chunk_size = max_seq_length

        X_chunked = torch.split(X_padded_batch, chunk_size, dim=1)

        if y != []:
            y_sequences = [torch.from_numpy(y[seq[0]:seq[1]]) for seq in b]
            y_padded_sequences = [torch.cat([t, -1*torch.ones(max_seq_length - len(t), 1).long()]) for t in y_sequences]
            y_padded_batch = torch.stack(y_padded_sequences)
            if use_cuda:
                y_padded_batch = y_padded_batch.cuda()

            y_chunked = torch.split(y_padded_batch, chunk_size, dim=1)
            yield X_chunked, y_chunked, b
        else:
            yield X_chunked, [], b # return [] (and not y) in case y is changed to some other "not available" value


# TODO @Future: collect_ensembles_preds was added in a hurry; can be beautified.
def get_indexed_predictions_with_targets(model, inputs, targets, sequence_bounds, cuda=False, collect_ensembles_preds=False):
    """
    Computes predictions for a model or ensemble of models on data according to sequence bounds.
    :param model: model, can also be a list of models working as an ensemble
    :param inputs: input data
    :param targets: golden labels
    :param sequence_bounds: list of sequences of given data for prediction
    :return: list of predictions (sorted); gold labels; mention indices
    """

    # For convenience during evaluation, this function will output token indices along with predictions and targets:
    token_indices = np.arange(len(targets), dtype=int)
    # Restrict to indices/targets for the given sequence bounds (subset of scenes/episodes to be processed):
    token_indices_subset = np.concatenate([token_indices[bounds[0]:bounds[1]] for bounds in sequence_bounds[sequence_bounds[:,0].argsort()]])
    targets_subset = targets[token_indices_subset]
    # And finally restrict to indices that are actual mentions:
    targets_subset_filtered = targets_subset[targets_subset != NON_ENTITY_MENTION]
    token_indices_subset_filtered = token_indices_subset[targets_subset != NON_ENTITY_MENTION]

    # Set model or models to eval() mode (switches off dropout, a.o.).
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()

    # TODO @Future: Take batch and chunk size from config file; currently it may run out of memory when testing (with cuda) on training set.
    predictions, raw_scores = get_all_predictions_and_mean_loss(model, inputs, targets, sequence_bounds, batch_size=len(inputs), chunk_size=-1, loss_function=None, use_cuda=cuda, collect_ensembles_preds=collect_ensembles_preds)

    if cuda: 
        predictions = predictions.cpu()
        if raw_scores != None:
            raw_scores = raw_scores.cpu()
    if not isinstance(predictions, dict):
        predictions = predictions.numpy()
        if raw_scores != None:
            raw_scores = raw_scores.numpy()
    
    return list(zip(token_indices_subset_filtered, predictions, targets_subset_filtered)), raw_scores


def get_all_predictions_and_mean_loss(model, X_data, y_data, sequence_bounds, batch_size, chunk_size=-1, loss_function=None, use_cuda=False, collect_ensembles_preds=False):
    """
    Computes a list of predictions (ordered according to original data) and mean loss over sequences specified by sequence bounds.
    With Volatile = True and returns a float, not a Variable -- hence not for backprop.
    :param model: model can also be an ensemble, i.e., a list of models.
    :param X_data: Numpy array of ALL inputs, in order of original data.
    :param y_data: Numpy array of ALL targets, in order of original data.
    :param sequence_bounds: Bounds of scenes/episodes, can be unsorted.
    :param batch_size: How many sequences (scenes/episodes) to feed to the model at once.
    :param chunk_size: How large parts of sequences to feed to the model at once.
    :param loss_function: E.g., NLLLoss; if None then no loss is computed (-1 is returned).
    :param use_cuda:
    :param grad: Whether to keep track of grad_fn (for gradients) along the way.
    :return: Tensor of predictions over sequence_bounds, sorted; Mean loss over sequence bounds, in a Variable only if volatile is false.
    """
    # Results of interest:
    total_loss = 0
    all_predictions = {}
    all_predictions_models = {}
    all_scores_models = {}

    for X_batch, y_batch, bounds in yield_padded_batch_and_bounds(X_data, y_data, sequence_bounds, batch_size, chunk_size, shuffle_data=False, use_cuda=use_cuda):

        # This was a bug once; it was reset every chunk.
        if isinstance(model, list):
            for m in model:
                m.eval()
                m.init_hidden(len(X_batch[0]))
        else:
            model.eval()
            model.init_hidden(len(X_batch[0]))

        for X_batch_chunk, y_batch_chunk in zip(X_batch, y_batch):
            # Compute desired outputs mask and create input variable:
            mask = (y_batch_chunk != NON_ENTITY_MENTION).squeeze()
            padded_batch = autograd.Variable(X_batch_chunk, volatile=True)  # Volatile=True means requires_grad=False.

            # Apply the model(s) to the input variable given the mask; outputs will be unpadded list of tensors.
            if isinstance(model, list):
                outputs = _apply_models_ensemble(model, padded_batch, desired_outputs_mask=mask, use_cuda=use_cuda, collect_ensembles_preds=collect_ensembles_preds)
            else:
                outputs = model(padded_batch, desired_outputs_mask=mask)
            # print(outputs)
            # Loop through all nonempty outputs, for each sequence in the batch, and store them
            # in the right place: in the dictionary all_predictions indexed by starting bound.
            num_predictions = 0
            # Apparently this explicit non-emptiness check is necessary (PyTorch bug in Variable's iter() perhaps?
            if len(outputs) > 0:
                if not collect_ensembles_preds:
                    for i, output in enumerate(outputs):
                        # If any predictions were made for this sequence
                        if len(outputs[i]) > 0:
                            _, predictions = torch.max(output.data, 1)
                            num_predictions += len(predictions)
                            # Either insert them anew or concatenate them with previous predictions for the sequence:
                            if bounds[i][0] not in all_predictions:
                                all_predictions[bounds[i][0]] = predictions
                            else:
                                all_predictions[bounds[i][0]] = torch.cat([all_predictions[bounds[i][0]], predictions])
                else:
                    models_outputs = outputs
                    for model_k, outputs_k in enumerate(models_outputs):
                        all_predictions_k = all_predictions_models.setdefault(model_k, {})
                        all_scores_models_k = all_scores_models.setdefault(model_k, {})
                        for i, output in enumerate(outputs_k):
                            # If any predictions were made for this sequence
                            if len(outputs_k[i]) > 0:
                                max_scores, predictions = torch.max(output.data, 1)
                                num_predictions += len(predictions)
                                # Either insert them anew or concatenate them with previous predictions for the sequence:
                                if bounds[i][0] not in all_predictions_k:
                                    all_predictions_k[bounds[i][0]] = predictions
                                    all_scores_models_k[bounds[i][0]] = max_scores
                                else:
                                    all_predictions_k[bounds[i][0]] = torch.cat([all_predictions_k[bounds[i][0]], predictions])
                                    all_scores_models_k[bounds[i][0]] = torch.cat([all_predictions_k[bounds[i][0]], max_scores])

            # Total_loss is weighed by number of predictions (this wasn't done in previous versions -- slight inaccuracy)
            if loss_function is not None:
                total_loss += num_predictions * _loss_from_outputs_batch(loss_function, outputs, y_batch_chunk)

    if not collect_ensembles_preds:
        # Sort the predictions; compute the mean loss
        all_predictions_sorted = torch.cat([all_predictions[k] for k in sorted(all_predictions.keys())], dim=0)
        all_scores_models_sorted = None
    else:
        all_predictions_sorted = {}
        all_scores_models_sorted = {}
        for model_k in all_predictions_models:
            all_predictions_sorted[model_k] = torch.cat([all_predictions_models[model_k][key] for key in sorted(all_predictions_models[model_k].keys())], dim=0)
            all_scores_models_sorted[model_k] = torch.cat([all_scores_models[model_k][key] for key in sorted(all_scores_models[model_k].keys())], dim=0)
    if loss_function is None:
        return all_predictions_sorted, all_scores_models_sorted

    # Else, namely, if loss_function is not None:
    mean_loss = total_loss / len(all_predictions_sorted)
    mean_loss = mean_loss.data[0]   # Take out of Variable, hence not for subsequent backprop.

    return all_predictions_sorted, mean_loss


def _loss_from_outputs_batch(loss_function, outputs_per_chunk, targets_per_chunk):
    """
    Applies the loss function to batch of outputs (unpadded, masked) and batch of targets (padded, unmasked).
    :param loss_function: e.g., NLLLoss.
    :param outputs_per_chunk: a list of tensors, each the relevant (i.e., masked) outputs for a chunk.
    :param targets_per_chunk: the UNMASKED targets (e.g., an y_batch_chunk given by yield_batches).
    :return: a Variable containing the loss.
    """

    # This nonempty-filter is probably no longer necessary in pytorch version 4 (bug in torch.cat).
    nonempty_outputs = [output for output in outputs_per_chunk if len(output) > 0]

    # But the following still is necessary, because the loss function cannot handle an empty tensor...
    if len(nonempty_outputs) == 0:
        loss = torch.Tensor([0.0])
        if outputs_per_chunk[0].is_cuda: loss = loss.cuda()
        return autograd.Variable(loss, requires_grad=True)

    # For computing the loss, first concatenate all the outputs:
    outputs_concatenated = torch.cat(nonempty_outputs, dim=0)  # Would give an error for empty tensors in PyTorch 3
    # And likewise 'pack' all targets into a long list and wrap inside a Variable:
    mask = (targets_per_chunk != NON_ENTITY_MENTION).squeeze()
    targets_packed, _ = tensor_utils.pack_batch_masked(targets_per_chunk, mask=mask)
    targets_packed = autograd.Variable(targets_packed, requires_grad=False).squeeze()

    # Easy peasy!
    loss = loss_function(outputs_concatenated, targets_packed)

    return loss


def _apply_models_ensemble(models, padded_batch, desired_outputs_mask, use_cuda=False, collect_ensembles_preds=False):
    """
    Takes a list of models and computes the average of their raw scores, on a padded batch, with a desired outputs mask.
    :param models:
    :param padded_batch:
    :param desired_outputs_mask:
    :param use_cuda:
    :return: a list of unpadded tensors, just like what forward() returns for a single model.
    """
    # Sum the outputs of all models:
    ensemble_outputs = None
    models_scores = []
    for model in models:
        model_outputs = model(padded_batch, desired_outputs_mask=desired_outputs_mask)
        models_scores.append([t for t in model_outputs])
        if ensemble_outputs is None:
            ensemble_outputs = [t for t in model_outputs]
        else:
            for i in range(len(model_outputs)):
                ensemble_outputs[i] = ensemble_outputs[i] + model_outputs[i]

    if collect_ensembles_preds == True:
        return models_scores
    else:
        # Divide by the number of models, for the sake of consistency:
        for i in range(len(ensemble_outputs)):
            ensemble_outputs[i] = ensemble_outputs[i] / float(len(models))
        return ensemble_outputs


# TODO @Future: include in logs precision and recall.
def train(model, X, y, train_sequence_bounds, validation_sequence_bounds, args, no_shuffle, logger):
    """
    Trains a model on training data X, y, testing on validation data, 
    with various settings given by args. If no validation set is given, 
    evaluates only on training data.
    :param model:
    :param X: Numpy array containing inputs (in original order)
    :param y: Numpy array containing targets (in original order, with NON-ENTITY_MENTION for non-mentions)
    :param train_sequence_bounds: List of pairs of [start, end] delineating training scenes/episodes in X/y.
    :param validation_sequence_bounds: List of pairs of [start, end] delineating validation scenes/episodes in X/y.
    :param args: forwarded from the argument parser in main.py.
    :return:
    """

    # Store arguments in convenient variables
    shuffle_data = not no_shuffle
    use_validation_data = len(validation_sequence_bounds)
    use_cuda = next(model.parameters()).is_cuda

    # This will be used later on to assess the model's performance
    if use_validation_data:
        y_val = np.concatenate([y[bounds[0]:bounds[1]] for bounds in validation_sequence_bounds[validation_sequence_bounds[:,0].argsort()]])
        answers_val = y_val[y_val != NON_ENTITY_MENTION]
    y_train =  np.concatenate([y[bounds[0]:bounds[1]] for bounds in train_sequence_bounds[train_sequence_bounds[:,0].argsort()]])
    answers_train = y_train[y_train != NON_ENTITY_MENTION]

    # Setup outputs if desired
    #model_file = None if args.no_files else os.path.join('models', file_name + '.pt')
    #log_file = sys.stdout if args.no_files else os.path.join('logs', file_name + '.log')

    # Define loss function and optimizer
    # TODO @Future: want to try max-margin loss as well.
    if args.class_weights is None:
        loss_function = nn.NLLLoss(ignore_index=NON_ENTITY_MENTION)  # non-mentions are labelled as NON_ENTITY_MENTION
    else:
        class_weights = torch.from_numpy(args.class_weights).float()
        if use_cuda: class_weights = class_weights.cuda()
        loss_function = nn.NLLLoss(weight=class_weights, ignore_index=NON_ENTITY_MENTION)

    params = filter(lambda p: p.requires_grad, model.parameters())
    # TODO @Future: improve if more (e.g., AdaGrad) optimizers want to be tried
    # TODO @Future: Apparently some have argued that weight decay shouldn't apply to embeddings... maybe try this?
    if args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    #elif args.useSGD: optimizer = optim.SGD(params, lr=args.lr) # TODO specify momentum?

    # Variables for bookkeeping during training
    epoch = -1  # Start at -1 in order to compute pre-training scores.
    iteration = 0
    min_val_loss = 1000.0
    max_val_macro_f1 = -1.0
    best_model = model
    prev_val_score = 0.0
    prev_train_score = 0.0
    num_epochs_no_improve = 0
    stop_criterion_data = "validation" if use_validation_data else "training"

    while epoch < args.epochs:

        logger.say('Epoch: '+ str(epoch))

        if epoch != -1:

            model.train()

            # Iterate through batches (lists of batches of chunks), don't care about the order (hence the '_')
            for X_batch, y_batch, _ in yield_padded_batch_and_bounds(X, y, train_sequence_bounds, args.batch_size, args.chunk_size, shuffle_data, use_cuda=use_cuda):

                #Reset hidden state for batch (reset for all three scenes/episodes)
                model.init_hidden(len(X_batch[0]))

                # These are the iterations (i.e., optimizer steps):
                for X_batch_chunk, y_batch_chunk in zip(X_batch, y_batch):

                    # Truncate backpropagation, back to start of chunk
                    model.detach_hidden()

                    # Wrap inputs in a variable; compute mask of where outputs are desired
                    padded_batch = autograd.Variable(X_batch_chunk)
                    mask = (y_batch_chunk != NON_ENTITY_MENTION).squeeze()

                    # Apply model and calculate loss
                    outputs = model(padded_batch, desired_outputs_mask=mask)
                    loss = _loss_from_outputs_batch(loss_function, outputs, y_batch_chunk)

                    # Backward step; retain graph for the next batch, which may continue the current sequence.
                    optimizer.zero_grad()
                    loss.backward(retain_graph=False)    # retain_graph is powerless anyway given that we truncate backpropagation now.
                    optimizer.step()

                    iteration += 1

        # Every N epochs, collect performance statistics:
        if (epoch + 1) % args.test_every == 0:

            model.eval()

            # Reserve a dictionary to store the various results in
            perf_measures = {"loss": -1, "total": -1, "accuracy": -1,
                             "macro_f1_score": -1, "macro_f1_score_main": -1, 'f1_scores': None}
            performance = {"epoch": epoch, "iteration": iteration,
                           "training": perf_measures,
                           "validation": perf_measures.copy()}

            # Obtain predictions and loss on training data; predictions are in order.
            train_predictions, mean_train_loss = get_all_predictions_and_mean_loss(
                model, X, y, train_sequence_bounds,
                batch_size=args.batch_size, chunk_size=args.chunk_size, loss_function=loss_function,
                use_cuda=use_cuda)
            if use_cuda: train_predictions = train_predictions.cpu()
            train_predictions = train_predictions.numpy()

            # Get all scores and insert them into the dictionary
            train_scores = get_scores(train_predictions, answers_train)
            performance["training"].update(train_scores)
            performance["training"]["loss"] = mean_train_loss

            # Also keep track of the relative performance increase/decrease:
            f1_diff_train = performance["training"]["macro_f1_score"] - prev_train_score
            prev_train_score = performance["training"]["macro_f1_score"]

            # To avoid None error during training if without crossvalidation (no validation data).
            mean_val_loss = mean_train_loss

            # And do the same for validation data:
            if use_validation_data:
                val_predictions, mean_val_loss = get_all_predictions_and_mean_loss(
                    model, X, y, validation_sequence_bounds,
                    batch_size=args.batch_size, chunk_size=args.chunk_size, loss_function=loss_function,
                    use_cuda=use_cuda)
                if use_cuda: val_predictions = val_predictions.cpu()
                val_predictions = val_predictions.numpy()

                val_scores = get_scores(val_predictions, answers_val)
                performance["validation"].update(val_scores)
                performance["validation"]["loss"] = mean_val_loss

            f1_diff_val = performance[stop_criterion_data]["macro_f1_score"] - prev_val_score
            prev_val_score = performance[stop_criterion_data]["macro_f1_score"]

            # Print the various scores
            logger.say('Mean loss: \n  training: {0:12.7f}\n  validation: {1:10.7f}'.format(mean_train_loss, mean_val_loss))
            logger.say('Accuracy: \n  training: {0[training][accuracy]:12.4f} (total {0[training][total]:7d})\n  validation: {0[validation][accuracy]:10.4f} (total {0[validation][total]:7d})'.format(performance))
            logger.say('Macro F1: \n  training: {0[training][macro_f1_score]:12.4f}   dif.: {1:.5f}   (total {0[training][total]:7d})\n  validation: {0[validation][macro_f1_score]:10.4f}   dif.: {2:.5f}   (total {0[validation][total]:7d})\n'.format(performance, f1_diff_train, f1_diff_val))

            # Keep track of best performance, and assess whether to stop training
            if mean_val_loss < min_val_loss or performance[stop_criterion_data]["macro_f1_score"] > max_val_macro_f1:    # i.e., if the model is improving.
                num_epochs_no_improve = 0
                best_model = model
                min_val_loss = min(mean_val_loss, min_val_loss)
                max_val_macro_f1 = max(performance[stop_criterion_data]["macro_f1_score"], max_val_macro_f1)
            else:                               # i.e., if the model is NOT improving.
                num_epochs_no_improve += args.test_every
                if num_epochs_no_improve >= args.stop_criterion:
                    message = 'Stopped after epoch {0} because validation loss did not decrease for {1} epochs.'.format(epoch, num_epochs_no_improve)
                    logger.say(message)
                    logger.log('# '+message)
                    break

        logger.log(performance)

        epoch += 1

    return model, best_model


def get_scores(predicted, targets, restrict_indices=None):
    """
    Computes prediction accuracy, F1 scores for all/main entities, macro average of F1 scores for all/main entities.
    :param predicted: list of predicted labels
    :param targets: list of true labels
    :param restrict_indices: numpy array containing indices (if any) to restrict the scores to (e.g., only pronouns).
    :return: dictionary of all scores
    """
    # TODO @Future: More of this could be done on gpu (e.g.: https://www.kaggle.com/igormq/f-beta-score-for-pytorch/code )

    all_scores = {}

    if restrict_indices is not None:
        # Restrict evaluation to a subset of the data (e.g., only pronouns)
        targets = targets[restrict_indices]
        predicted = predicted[restrict_indices]

    for all_entities in [True, False]:

        if not all_entities:
            # Transform the predictions and targets to include only main entities and 'other'
            # NOTE: simply setting labels=MAIN_ENTITIES in sklearn's F1-score function is not equivalent!
            predicted = predicted.copy()
            predicted[~np.isin(predicted, MAIN_ENTITIES)] = OTHER_ENTITY
            targets = targets.copy()
            targets[~np.isin(targets, MAIN_ENTITIES)] = OTHER_ENTITY

        classes = list(set(targets))    # Compute (average) F1 scores only for entities that occur in targets.

        # Compute scores using sklearn:
        warnings.filterwarnings("ignore")
        f1_scores_class = f1_score(targets, predicted, labels=classes, average=None) * 100
        accuracy = accuracy_score(targets, predicted) * 100
        macro_f1_score = np.mean(f1_scores_class)

        # Return appropriately named scores:
        add_suffix = lambda s: (s if all_entities else s+"_main")
        all_scores.update({add_suffix('accuracy'): accuracy,
                           add_suffix('macro_f1_score'): macro_f1_score,
                           add_suffix('total'): len(targets) if all_entities else sum(targets != OTHER_ENTITY),
                           })

    return all_scores
