import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import tensor_utils

"""
Provides classes LSTM_basic, UtteranceEmbedder, Attender, StaticEntitySimilarizer
"""

class LSTM_basic(nn.Module):

    def __init__(self, args, padding_idx=-1):
        """

        :param args: Namespace containing at least the parsed 'model' section of the config file.
        :param padding_idx: padding is used in several places for memory efficiency
        """

        super(LSTM_basic, self).__init__()

        self.padding_idx = padding_idx

        # Input layers:
        input_to_lstm = []
        # emb_reduction = [0,400] # TODO @Future: implement and add as config option.
        embedder = UtteranceEmbedder(args, padding_idx=padding_idx)
        input_to_lstm.append(embedder)
        if args.dropout_prob_1 > 0.0:       input_to_lstm.append(torch.nn.Dropout(p=args.dropout_prob_1))
        if args.nonlinearity_1 == 'tanh':   input_to_lstm.append(torch.nn.Tanh())
        elif args.nonlinearity_1 == 'relu': input_to_lstm.append(torch.nn.ReLU())
        self.input_to_lstm = nn.Sequential(*input_to_lstm)

        # LSTM:
        self.lstm = nn.LSTM(embedder.embedding_dim,
                            args.hidden_lstm_1 // (2 if args.bidirectional else 1),
                            num_layers=args.layers_lstm,
                            batch_first=True,
                            bidirectional=args.bidirectional)

        # Apply attention over LSTM's outputs
        if isinstance(args.attention_lstm, str):
            self.attention_lstm = Attender(embedder.embedding_dim,
                                           self.lstm.hidden_size * (2 if args.bidirectional else 1),
                                           args.attention_lstm, args.nonlinearity_a,
                                            attention_window = args.attention_window,
                                           window_size = args.window_size)
        else:
            self.attention_lstm = None      # For easy checking in forward()

        # Output layers:
        lstm_to_output = []
        if args.dropout_prob_2 > 0.0: lstm_to_output.append(torch.nn.Dropout(p=args.dropout_prob_2))

        if not args.entity_library:
            lstm_to_output.append(nn.Linear(self.lstm.hidden_size*(2 if args.bidirectional else 1),
                                            args.num_entities))
        else:
            lstm_to_output.append(nn.Linear(self.lstm.hidden_size*(2 if args.bidirectional else 1),
                                            embedder.speaker_emb.weight.data.shape[1]))
        if args.nonlinearity_2 == 'tanh':   lstm_to_output.append(torch.nn.Tanh())
        elif args.nonlinearity_2 == 'relu': lstm_to_output.append(torch.nn.ReLU())
        if args.entity_library:
            lstm_to_output.append(StaticEntitySimilarizer(embedder.speaker_emb,
                                                          args.similarity_type,
                                                          share_weights=args.share_weights))
        lstm_to_output.append(nn.LogSoftmax(dim=-1))
        self.lstm_to_output = nn.Sequential(*lstm_to_output)

    def init_hidden(self, batch_size):
        """
        Resets the LSTM's hidden layer activations. To be called before applying the LSTM to a batch (not chunk).
        With batch_first True, the axes of the hidden layer are (minibatch_size, num_hidden_layers, hidden_dim).
        :param batch_size:
        """
        hidden1 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1),
                              batch_size,
                              self.lstm.hidden_size)
        # Hidden state will be cuda if whatever first parameter is.
        if next(self.parameters()).is_cuda: hidden1 = hidden1.cuda()
        hidden2 = torch.zeros_like(hidden1)
        self.hidden = (autograd.Variable(hidden1), autograd.Variable(hidden2))

    def detach_hidden(self):
        """
        This function is called to truncate backpropagation (e.g., at the start of each chunk).
        By wrapping the hidden states in a new variable it resets the grad_fn history for gradient computation.
        :return: nothing
        """
        self.hidden = tuple([autograd.Variable(hidden.data) for hidden in self.hidden])

    def forward(self, padded_batch, desired_outputs_mask=None):
        """
        Applies the model to a padded batch of chunked sequences.
        :param padded_batch: padded batch of shape batch_size x chunk_size x chunk_length.
        :param desired_outputs_mask: boolean mask (batch size x chunk size) of which rows in each
                chunk require an output (entity mention).
        :return: softmaxed scores, as a list of tensors of varying lengths if mask is given; padded tensor with all outputs otherwise.
        """
        # For backwards compatibility, used only for TESTING:
        if not isinstance(padded_batch, autograd.Variable):
            padded_batch = autograd.Variable(padded_batch)

        embeddings = self.input_to_lstm(padded_batch)

        # TODO @Future: At some point we should use pack_padded etc; then maybe nicer if input to forward() is a list of unpadded sequences.
        # embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, chunk_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # lstm_out, lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # If necessary apply attention to lstm's outputs:
        if self.attention_lstm is not None:
            lstm_out = self.attention_lstm(embeddings, lstm_out)

        # If a mask was given, continue only with the outputs of certain rows (per chunk):
        if desired_outputs_mask is not None:
            # Empty mask means we can already stop right away.
            if desired_outputs_mask.sum() == 0:
                # Return list of num_chunk empty tensors.
                empty_tensor = torch.FloatTensor()
                if padded_batch.is_cuda: empty_tensor = empty_tensor.cuda()
                empty_tensor = autograd.Variable(empty_tensor)
                output = [empty_tensor.clone() for _ in padded_batch]
                return output
            # Else (i.e., non-empty mask):
            lstm_out, lengths = tensor_utils.pack_batch_masked(lstm_out, desired_outputs_mask)

        output = self.lstm_to_output(lstm_out)

        # If it was packed, make sure to unpack it afterwards (without repadding):
        if desired_outputs_mask is not None:
            output = tensor_utils.unpack_packed_batch(output, lengths)

        return output


class UtteranceEmbedder(nn.Module):

    def __init__(self, args, padding_idx=-1):
        super(UtteranceEmbedder, self).__init__()

        self.padding_idx = padding_idx

        input_dims = [args.vocabulary_size, args.num_entities]
        embeddings_specs = [args.token_emb, args.speaker_emb]

        # Token embeddings, either pre-trained or random weights
        if isinstance(embeddings_specs[0], np.ndarray):
            emb_weights = torch.Tensor(embeddings_specs[0])
            self.token_emb = nn.Embedding(emb_weights.shape[0], emb_weights.shape[1])
            self.token_emb.weight.data = emb_weights
        else:
            self.token_emb = nn.Embedding(input_dims[0], embeddings_specs[0])            

        # Speaker embeddings, either pre-trained or random weights
        if isinstance(embeddings_specs[1], np.ndarray):
            emb_weights = torch.Tensor(embeddings_specs[1])
            self.speaker_emb = nn.EmbeddingBag(emb_weights.shape[0], emb_weights.shape[1])
            self.speaker_emb.weight.data = emb_weights
        else:
            self.speaker_emb = nn.EmbeddingBag(input_dims[1], embeddings_specs[1], mode='sum')

        self.embedding_dim = self.token_emb.embedding_dim + self.speaker_emb.embedding_dim

    def forward(self, padded_batch):

        # Compute embeddings of input
        token_embeddings = self._padded_embedding(self.token_emb, padded_batch[:, :, 0])
        speaker_embeddings = self._padded_embedding(self.speaker_emb, padded_batch[:, :, 1:])
        
        embeddings = torch.cat((token_embeddings, speaker_embeddings), 2)

        return embeddings

    def _padded_embedding(self, emb, ids):
        """
        This wrapper gets rid of padding indices prior to computing embeddings, returning all-zeros.
        This was necessary because PyTorch's Embedding cannot (yet) handle padding_idx = -1, and because
        EmbeddingBag cannot (yet?) handle padding_idx at all.
        :param emb: Embedding or EmbeddingBag
        :param ids: BxS (for Embedding) or BxSxN (for EmbeddingBag)
        :return:
        """
        # First remove all padding indices to obtain the actual ids
        mask = ids != self.padding_idx
        actual_ids = torch.masked_select(ids, mask)

        # Prepare for, and feed through, EmbeddingBag (3D) or Embedding (2D)
        if len(ids.shape) == 3:
            # If 3D tensor of indices, prepare them for EmbeddingBag
            sum_ids = mask.long().sum(2)
            sum_ids = torch.masked_select(sum_ids, mask[:, :, 0])
            cumsum_speakers = sum_ids.cumsum(0)
            offsets = torch.zeros_like(cumsum_speakers)
            if len(offsets) > 1:  # Necessary to avoid slice yielding empty tensor, which apparently is forbidden.
                offsets[1:] = cumsum_speakers[:-1]
            # Now that they're a long 1D tensor, compute their embeddings
            actual_embs = emb(actual_ids, offsets)
            # Compute a mask to put them back together in a new tensor below
            embedding_mask = mask[:, :, 0].contiguous().view(
                ids.shape[0], ids.shape[1], 1).expand(-1, -1, emb.weight.shape[1])
        else:
            # Else, assuming 2D tensor of indices, feed them through plain Embedding
            actual_embs = emb(actual_ids)
            # Compute a mask to put the results back together in a new tensor below
            embedding_mask = mask.view(ids.shape[0], ids.shape[1], 1).expand(-1, -1, emb.weight.shape[1])

        # Tensor that will hold the embeddings (batch size x chunk length x embedding dim) amidst zeros
        embeddings = torch.zeros(ids.shape[0], ids.shape[1], emb.weight.shape[1])
        if ids.is_cuda: embeddings = embeddings.cuda()
        embeddings = autograd.Variable(embeddings)
        # Scatter the computed embeddings into the right places in the new tensor
        embeddings.masked_scatter_(embedding_mask, actual_embs)

        return embeddings


class StaticEntitySimilarizer(nn.Module):

    def __init__(self, speaker_embedder, sim='dot', share_weights = True):
        super(StaticEntitySimilarizer, self).__init__()

        if share_weights:
            self.entity_embedder = speaker_embedder
        else:
            emb_weights = speaker_embedder.weight.data.clone()
            self.entity_embedder = nn.EmbeddingBag(emb_weights.shape[0], emb_weights.shape[1], mode= "sum")
            self.entity_embedder.weight.data = emb_weights

        # Similarity functions produce output of dim BxCxN or (B*C)xN, depending on whether input was packed
        if sim == 'dot':
            # x1: BxCxM or BxM    x2: MxN (= NxM transposed)
            self.similarity_function = lambda x1, x2: torch.matmul(x1, x2.t())
        elif sim == 'cos':
            # x1: BxCx1xM or Bx1xM  x2: 1x1xNxM or 1xNxM
            cos = nn.CosineSimilarity(dim=-1)
            self.similarity_function = lambda x1, x2: cos(x1.unsqueeze(-2), x2.unsqueeze(0).unsqueeze(0) if len(x1.shape) == 3 else x2.unsqueeze(0))

    def forward(self, padded_batch):

        # Apply the chosen similarity function ==> output dim: BxCxN
        output = self.similarity_function(padded_batch, self.entity_embedder.weight)

        # The following is an ugly fix, necessary because cosine sim makes the chunk
        # size dimension disappear if the chunk size is 1. PyTorch Bug?
        if len(padded_batch.shape) > len(output.shape):
            output = output.unsqueeze(-2)

        return output


class Attender(nn.Module):

    def __init__(self, query_dim, key_dim, attention_type, nonlinearity,
                 attention_window = True, window_size = 20, max_chunk_size=999):
        super(Attender, self).__init__()
        self.attention_type = attention_type

        self.attention_window = None
        if attention_window:
            self.attention_window = self._get_attention_tensor_with_window_size(window_size, max_chunk_size)

        if self.attention_type == 'feedforward':
            if nonlinearity == 'tanh':
                nonlin_activation = torch.nn.Tanh()
                self.attention_layer = nn.Sequential(nn.Linear(query_dim + key_dim, 1),
                                                     nonlin_activation)
            elif nonlinearity == 'relu':
                nonlin_activation = torch.nn.ReLU()
                self.attention_layer = nn.Sequential(nn.Linear(query_dim + key_dim, 1),
                                                 nonlin_activation)
            else:
                self.attention_layer = nn.Linear(query_dim + key_dim, 1)

        elif self.attention_type == 'dot':
            if key_dim >= query_dim:
                self.reduce = nn.Linear(key_dim, query_dim)
                self.match_dims = lambda queries, keys: (queries, self.reduce(keys))
            else:
                self.reduce = nn.Linear(query_dim, key_dim)
                self.match_dims = lambda queries, keys: (self.reduce(queries), keys)

    def forward(self, queries, keys, values=None):

        debug = False

        # If no separate values are given, do self-attention:
        if values == None:
            values = keys
        if debug: print("IN: Queries:", queries.shape, "Keys:", keys.shape)

        chunk_size = keys.size()[1]
        batch_size = keys.size()[0]
        
        if self.attention_type == 'feedforward':
            similarities = torch.Tensor(batch_size, chunk_size, chunk_size)
            if queries.is_cuda: similarities = similarities.cuda()
            similarities = autograd.Variable(similarities)
            # Compute similarities one chunk at a time, otherwise we risk out of memory error :(
            for i in range(0, batch_size, 1):
                some_queries = queries[i:i+1].unsqueeze(2).expand(-1, -1, chunk_size, -1)
                some_keys = keys[i:i+1].unsqueeze(1).expand(-1, chunk_size, -1, -1)
                pairs = torch.cat((some_queries, some_keys), dim=-1)
                similarities[i:i+1] = self.attention_layer(pairs).view(1, chunk_size, chunk_size)
        elif self.attention_type == 'dot':
            queries, keys = self.match_dims(queries, keys)
            similarities = torch.bmm(queries, keys.transpose(-2,-1))
            #                        Bx(C1xD) @  Bx(DxC2)     = Bx(C1xC2)

        if self.attention_window is not None:
            if not self.attention_window.is_cuda and queries.is_cuda:
                # Will execute just once:
                self.attention_window = self.attention_window.cuda()
            similarities = similarities + self.attention_window[0:chunk_size, 0:chunk_size]

        # For every query (C1), similarities to all its keys (C2) must sum to 1 -- that's the last axis (-1)
        weights = F.softmax(similarities, dim=-1)

        # Multiply the values by the weights
        weighted_values = torch.bmm(weights, values)
        #                          Bx(C1xC2) @ Bx(C2xD) = Bx(C1xD)

        return weighted_values

    def _get_attention_tensor_with_window_size(self, window_size, chunk_size):
        """
        TODO @Future: This can be simplified to 2 or 3 lines of code.
        Computes something like 11000000
                                11100000
                                01110000
                                00111000
                                00011100
                                00001110
                                00000111
                                00000011
        :param window_size:
        :param chunk_size:
        :return:
        """
        dim_inner = chunk_size - 2 * window_size

        # Construct left part of tensor
        left_values = torch.Tensor([])
        number_zeros = window_size
        number_negatives = chunk_size - window_size
        for i in range(1, window_size + 1):
            number_zeros += 1
            number_negatives -= 1
            left_values = torch.cat((left_values, torch.zeros(number_zeros),
                                     torch.from_numpy(np.array([-10 ** (8) for i in range(number_negatives)])).float()), dim=0)

        # Construct inner part of tensor
        number_zeros = 2 * window_size + 1
        number_negatives = chunk_size - number_zeros + 1
        column_inner = torch.cat((torch.zeros(number_zeros),
                                  torch.from_numpy(np.array([-10**(8) for i in range(number_negatives)])).float()), dim=0)
        inner_values = column_inner.unsqueeze(0).expand(dim_inner, -1).contiguous().view(-1)[:-number_negatives]

        # Construct right part of tensor
        right_values = torch.from_numpy(np.flip(left_values.numpy(), 0).copy())

        # Put everything together
        values_vector = torch.cat((left_values, inner_values, right_values), dim=0)
        matrix_values = values_vector.contiguous().view(chunk_size, chunk_size)

        matrix_values = autograd.Variable(matrix_values, requires_grad=False)

        return matrix_values