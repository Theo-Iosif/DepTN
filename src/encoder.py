# --- imports --- #

import numpy as np

from helper_functions import generate_haar_isometric

# --- global tensor cores for MPS encoder --- #

def tensor_core_bulk(inner_dim: int, max_bond: int, isometric: bool = True):
    '''
    Generates a random tensor core for MPS representation in the bulk.

    input: inner_dim (int): The dimension of the tensor core.
           max_bond (int): The maximum bond dimension for the MPS.

    output (np.ndarray): A randomly generated tensor core.
    '''

    if isometric:
        return generate_haar_isometric(max_bond, inner_dim) # need to make it unitary using QR decomposition
    else:
        return np.random.rand(max_bond, inner_dim, max_bond)

def input_tensor_core(input_dim: int, max_bond: int):
    '''
    Generates a random tensor core for MPS representation for the input.

    input: input_dim (int): The dimension of the input.
           max_bond (int): The maximum bond dimension for the MPS.

    output (np.ndarray): A randomly generated tensor core.
    '''

    return np.random.rand(max_bond, input_dim, max_bond) # doesn't need to be unitary, but needs to be regularized to avoid exploding gradients


def tensor_core_dict(input_dim: int, inner_dim: int, max_order: int, max_bond: int, isometric: bool = True):
    '''
    Generates a dictionary of random tensor cores for MPS representation.

    input: input_dim (int): The dimension of the input.
           inner_dim (int): The dimension of the inner tensor cores.
           max_order (int): The maximum order of the MPS representation.
           max_bond (int): The maximum bond dimension for the MPS.
           
    
    output (dict): A dictionary of randomly generated tensor cores.
    '''

    tensor_dict = {0: input_tensor_core(input_dim, max_bond)}
    for i in range(1, max_order+1):
        tensor_dict[i] = tensor_core_bulk(inner_dim, max_bond, isometric)

    return tensor_dict


# --- vector to tensor conversion --- #

def uniform_mps_encoder(tensor_core_key: int, order: int):
    '''
    Encodes a given vector into a Matrix Product State (MPS) representation 
    based on a single, repeating tensor core.

    input: tensor_core_key (int): The key for the tensor core to be used.
           order (int): The order of the MPS representation.

    output (list): An opt_einsum-style interleaved list of symbolic tensor keys,
        integer index lists, and final output indices. Tensor key 0 is the
        input core; tensor_core_key is reused for every bulk core.

        The first index of the result is the input leg and the remaining indices
        are the order output legs. The two MPS boundary bonds are closed together,
        so no bond indices remain in the result.
    '''

    if order < 1:
        raise ValueError("order must be at least 1")

    # Interleaved integer labels are accepted by opt_einsum.contract:
    # [operand, labels, operand, labels, ..., output_labels].
    input_labels = [0, 1, 2]
    einsum_list = [0, input_labels]
    output_labels = [1]

    for position in range(order):
        left_bond = 2 * position + 2
        output_label = 2 * position + 3
        right_bond = 2 * position + 4 if position < order - 1 else 0 # maybe change to %(2*order+2)
        einsum_list.extend([
            tensor_core_key,
            [left_bond, output_label, right_bond],
        ])
        output_labels.append(output_label)

    einsum_list.append(output_labels)
    return einsum_list


def general_mps_encoder(max_order: int, order: int):
    '''
    Encodes a given vector into a Matrix Product State (MPS) representation 
    based on various tensor cores.

    input: order (int): The order of the MPS representation.
           max_order (int): The maximum order of the MPS representation.

    output (list): An opt_einsum-style interleaved list of symbolic tensor keys,
        integer index lists, and final output indices. Tensor key 0 is the
        input core; tensor_core_key is reused for every bulk core.

        The first index of the result is the input leg and the remaining indices
        are the order output legs. The two MPS boundary bonds are closed together,
        so no bond indices remain in the result.
    '''

    if order > max_order:
        raise ValueError("max_order too small for requested order")
    if order < 1:
        raise ValueError("order must be at least 1")

    # Interleaved integer labels are accepted by opt_einsum.contract:
    # [operand, labels, operand, labels, ..., output_labels].
    input_labels = [0, 1, 2]
    einsum_list = [0, input_labels]
    output_labels = [1]

    for position in range(order):
        left_bond = 2 * position + 2
        output_label = 2 * position + 3
        right_bond = 2 * position + 4 if position < order - 1 else 0 # maybe change to %(2*order+2)
        einsum_list.extend([
            position + 1, # tensor core key for this position
            [left_bond, output_label, right_bond],
        ])
        output_labels.append(output_label)

    einsum_list.append(output_labels)
    return einsum_list


# open boundary conditions for MPS representation:

# def tensor_core_edge(dimension: int, max_bond: int):
#     '''
#     Generates a random tensor core for MPS representation at the edge.

#     input: dimension (int): The dimension of the tensor core.
#            max_bond (int): The maximum bond dimension for the MPS.

#     output (np.ndarray): A randomly generated tensor core.
#     '''

#     return np.random.rand(dimension, max_bond)

# def open_bounds(max_bond: int):
#     '''
#     Generates a random boundary vector for the MPS.

#     input: max_bond (int): The maximum bond dimension for the MPS.

#     output (np.ndarray): A randomly generated tensor core with open boundaries.
#     '''

#     return np.random.rand(max_bond)