import torch
import torch.nn.functional as F
import torch.autograd as autograd

"""
Contains some functionality for manipulating tensors, used primarily in models.py; 
primarily for masked packing/unpacking. Some of this bypasses PyTorch functionality
that wasn't implemented back in version 3, but may be now. 
"""

def pack_batch_masked(padded_batch, mask):
    """
    Packs a padded batch of size BxLxD, according to mask of size BxL,
    to size SxD where S is the sum of unpadded lengths of all sequences in batch.
    :param padded_batch: BxLxD
    :param mask: BxL
    :return: packed batch (tensor of shape SxD), list with lengths of packed sequences (summing to S).
    """
    # Compute number of desired outputs (entity mentions) in each chunk
    mask_lengths = list(mask.long().sum(dim=1))
    # Flatten batch and chunk size dimensions, to create one long tensor (lstm_out and mask).
    padded_batch = padded_batch.contiguous().view(-1, padded_batch.shape[2])
    mask = mask.view(-1)
    # use only the rows in lstm_out according to the mask:
    mask_indices = mask.nonzero().squeeze()
    # The following check is necessary to avoid error, because Variables cannot be indexed with empty tensor :(
    if len(mask_indices) == 0:
        output = torch.FloatTensor()
        if padded_batch.is_cuda: output = output.cuda()
    else:
        output = padded_batch[mask_indices]
    return output, mask_lengths


def unpack_packed_batch(packed_batch, lengths, repad = False):
    """
    The reverse of pack_batch_masked: takes sequences according to mask_lengths out of a packed batch, optionally repadding them.
    :param packed_batch: tensor of shape SxD
    :param mask_lengths: list of lengths, summing to S
    :param repad: whether the unpacked list of sequences must be repadded into a single tensor.
    :return: list of sequences (tensors) of varying length, or a single tensor padding them.
    """
    unpacked_batch = split(packed_batch, lengths, dim=0)
    if repad:
        unpacked_batch = pad_sequences(unpacked_batch)        # Pad to length of longest mask
    return unpacked_batch


# Not used
def unpack_packed_batch_masked(packed_batch, mask_lengths, mask):
    """
    Like unpack_packed_batch, but scatters the resulting sequences back into a tensor of the original mask size.
    :param packed_batch:
    :param mask_lengths:
    :param mask:
    :return:
    """
    unpacked_batch = split(packed_batch, mask_lengths, dim=0)

    # Expand the mask to the new final dim size
    mask = autograd.Variable(mask.view(mask.shape[0], mask.shape[1], 1).expand(-1, -1, packed_batch.shape[-1]))

    # Create a zeros tensor the original mask size
    tensors = [torch.zeros(mask.shape[1], packed_batch.shape[-1]) for _ in range(mask.shape[0])]
    if unpacked_batch[0].is_cuda:
        tensors = [tensor.cuda() for tensor in tensors]
    tensors = [autograd.Variable(tensor) for tensor in tensors]

    # For each sequence, scatter the values back in place according to the mask
    for i, tensor in enumerate(tensors):
        tensor.masked_scatter_(mask[i], unpacked_batch[i])

    tensor = torch.stack(tensors)

    return tensor


# Not used.
def batch_masked_select(padded_batch, mask, repad=False):
    """
    From a padded tensor BxLxD of B sequences padded to length L,
    and mask of shape BxL, selects from each sequence elements
    where the mask is true, and returns this either as a list of tensors,
    or stacks these elements together in a new padded tensor of BxMxD of
    B sequences padded to length M.
    :param padded_batch: BxLxD
    :param mask: BxL
    :param repad: whether to return a list of tensors of varying length, or a single padded tensor.
    :return: list of tensors of varying length M, MxD, or tensor BxMxD where M is longest masked sequence length.
    """
    packed_batch, lengths = pack_batch_masked(padded_batch, mask)
    result = unpack_packed_batch(packed_batch, lengths, repad=repad)
    return result


def pad_sequences(list_of_tensors):
    """
    Takes a list of sequences of varying length, pads them to the max length, and stacks them into a single tensor.
    :param list_of_tensors: list of B tensors of shape LxD, of varying lengths L.
    :return: single tensor of shape Bx(max L)xD
    """
    seq_lengths = [t.shape[0] for t in list_of_tensors]
    max_seq_length = max(seq_lengths)
    padded_list = [F.pad(t, (0, 0, 0, max_seq_length - t.shape[0]), mode='constant', value=-1) for t in list_of_tensors]
    stacked = torch.stack(padded_list)
    return stacked


# TODO @Future: In PyTorch version 4.0, this may no longer be necessary, but part of torch.split(); except perhaps for the empty tensors fix? :(
def split(tensor, section_sizes, dim=0):
    # Splits a tensor into parts according to section_sizes

    split_indices = [0] + section_sizes
    split_indices = torch.cumsum(torch.Tensor(split_indices), dim=0)

    # because tensor.narrow doesn't like to create empty tensors:
    if 0 in section_sizes:
        empty_tensor = torch.Tensor()
        if tensor.is_cuda: empty_tensor = empty_tensor.cuda()
        if isinstance(tensor, autograd.Variable):
            empty_tensor = autograd.Variable(empty_tensor)

    return tuple(
        empty_tensor if length == 0 else tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(split_indices, section_sizes))

