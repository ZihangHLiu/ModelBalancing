import torch
import torch.nn as nn
import numpy as np
import math

def feature_alignment(prev_model, curr_model, eig_num=10, conv_norm=0.5, sample_num=None):
    """
    Summary

    This function is used to calculate the "angle" of two column spaces spanned by the largest eigenvectors of the model at two training steps.
    It first do SVD of both model weights, calculate the gram matrix of the principle(largest) eigenvectors of them, and then calculate the kernel alignment
    between two gram matrices.
    
    Args:
        prev_model: the model weight of the last saving point
        curr_model: the model weight of the current step
        eig_num: number of(largest) eigenvectors chosen to compute the alignment
        align_type (str): which alignment method to use
        sample_num (int, optional): number of pairs of eigenvectors to sample, use only when align_type == "min_sample"

    """
    results = {
        'longname':[],
        'kernel_alignment': [],
        'sample_alignment_min': [],
        'sample_alignment_mean': [],
        'sample_alignment': [],
    }
    for (name1, m_1), (name2, m_2) in zip(prev_model.named_modules(), curr_model.named_modules()):
        if isinstance(m_1, nn.Conv2d) or isinstance(m_1, nn.Linear):
            matrix_1 = m_1.weight.data.clone()
            matrix_2 = m_2.weight.data.clone()

            # normalization and tranpose Conv2d
            if isinstance(m_1, nn.Conv2d):
                matrix_1 = torch.flatten(matrix_1, start_dim=2) * math.sqrt(conv_norm)
                matrix_1 = matrix_1.transpose(1, 2).transpose(0, 1)
                matrix_2 = torch.flatten(matrix_2, start_dim=2) * math.sqrt(conv_norm)
                matrix_2 = matrix_2.transpose(1, 2).transpose(0, 1)

            U_1, S_1, Vh_1 = torch.linalg.svd(matrix_1)
            U_2, S_2, Vh_2 = torch.linalg.svd(matrix_2)
        
            # get kernel alignment
            # get gram matrices
            gram_1 = U_1[: eig_num, :] @ U_1[: eig_num, :].T
            gram_2 = U_2[: eig_num, :] @ U_2[: eig_num, :].T

            # calculate kernel alignment
            kernel_alignment = torch.sum(gram_1 * gram_2) / (torch.norm(gram_1) * torch.norm(gram_1))

            # get sample alignment
            assert sample_num > 0
            sample_align_list = []
            for i in range(sample_num):
                V_1 = U_1[: eig_num, :]
                V_2 = U_2[: eig_num, :]
                idx_1 = torch.randint(0, V_1.shape[0], (1,)).item()
                idx_2 = torch.randint(0, V_2.shape[0], (1,)).item()

                sample_alignment = torch.dot(V_1[idx_1], V_2[idx_2])
                sample_align_list.append(sample_alignment.cpu().item())

            sample_alignment_min = min(sample_align_list)
            sample_alignment_mean = np.mean(sample_align_list)

            results['longname'].append(name1)
            results['kernel_alignment'].append(kernel_alignment)
            results['sample_alignment_min'].append(sample_alignment_min)
            results['sample_alignment_mean'].append(sample_alignment_mean)
            results['sample_alignment'].append(sample_align_list)

    return results

