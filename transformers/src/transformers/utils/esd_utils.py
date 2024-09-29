import torch
import torch.nn as nn
from operator import itemgetter
import numpy as np
import math
import tqdm
import re
import os
from pyhessian import hessian

def net_esd_estimator(
            net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5,
            eigs_num_thresh=50, 
            filter_zeros=False,
            init_stable_rank=None,
            sr_mid_pos=None):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'stable_rank':[],
        'norm_stable_rank':[],
        'init_norm_stable_rank':[],
        'eig_ratio':[],
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone()
            # i have checked that the multiplication won't affect the weights value
            #print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()

            if len(eigs) < eigs_num_thresh:
                continue

            if filter_zeros:
                sr_eigs = eigs[eigs > EVALS_THRESH]
                if len(sr_eigs) == 0:
                    sr_eigs = eigs
            else:
                sr_eigs = eigs

            if sr_mid_pos is not None:
                mid = int(len(sr_eigs) / sr_mid_pos)
                sr_eigs = sr_eigs[mid: ]
            eigs_sum = torch.sum(sr_eigs)
            max_eigs = torch.max(sr_eigs)
            stable_rank = eigs_sum / max_eigs
            norm_stable_rank = eigs_sum / len(sr_eigs)
            mid_eig = sr_eigs[len(sr_eigs) // 2]
            eig_ratio = max_eigs / mid_eig

            results['stable_rank'].append(stable_rank.item())
            results['norm_stable_rank'].append(norm_stable_rank.item())
            if init_stable_rank is not None:
                results['init_norm_stable_rank'].append(norm_stable_rank.item() / init_stable_rank[len(results['init_norm_stable_rank'])])
            else:
                results['init_norm_stable_rank'].append(0)
            results['eig_ratio'].append(eig_ratio.item())
            
            if filter_zeros:
                #print(f"{name} Filter Zero")
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    #print(f"{name} No non-zero eigs, use original total eigs")
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                #print(f"{name} Skip Filter Zero")
                nz_eigs = eigs
                N = len(nz_eigs)

            nz_eigs = nz_eigs.to("cuda:0")
            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n).cuda()
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()

            results['spectral_norm'].append(spectral_norm)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())
    
    return results
        

def evals_esd_estimator(
            eigs_lst=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'name':[],
        'stable_rank':[],
        }
    # iterate through layers
    for eigs in eigs_lst:
        eigs, _ = torch.sort(eigs)
        spectral_norm = eigs[-1].item()
        results['spectral_norm'].append(spectral_norm)
        nz_eigs = eigs[eigs > EVALS_THRESH]

        N = len(nz_eigs)
        log_nz_eigs    = torch.log(nz_eigs)

        if fix_fingers == 'xmin_mid':
            i = len(nz_eigs) // xmin_pos
            xmin = nz_eigs[i]
            n = float(N - i)
            seq = torch.arange(n).cuda()
            final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            final_D = torch.max(torch.abs(
                        1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                    ))
        else:
            alphas = torch.zeros(N-1)
            Ds     = torch.ones(N-1)
            if fix_fingers == 'xmin_peak':
                hist_nz_eigs = torch.log10(nz_eigs)
                min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                boundaries = torch.linspace(min_e, max_e, bins + 1)
                h = counts, boundaries
                ih = torch.argmax(h[0])  # 
                xmin2 = 10 ** h[1][ih]
                xmin_min = torch.log10(0.95 * xmin2)
                xmin_max = 1.5 * xmin2
            
            for i, xmin in enumerate(nz_eigs[:-1]):
                if fix_fingers == 'xmin_peak':
                    if xmin < xmin_min:
                        continue
                    if xmin > xmin_max:
                        break

                n = float(N - i)
                seq = torch.arange(n).cuda()
                alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                alphas[i] = alpha
                if alpha > 1:
                    Ds[i] = torch.max(torch.abs(
                        1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                    ))

            min_D_index = torch.argmin(Ds)
            final_alpha = alphas[min_D_index]
            final_D = Ds[min_D_index]
        
        final_alpha = final_alpha.item()
        final_D = final_D.item()

        results['alpha'].append(final_alpha)
        results['D'].append(final_D)

    return results

def lora_esd_estimator(
            net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5,
            eigs_num_thresh=50, 
            filter_zeros=False,
            init_stable_rank=None,
            sr_mid_pos=None,
            add_W=True):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'stable_rank':[],
        'norm_stable_rank':[],
        'init_norm_stable_rank':[],
        'eig_ratio':[],
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        class_type_names = [n.__name__ for n in m.__class__.__bases__]
        if ("linear" in m.__class__.__name__.lower() or "conv" in m.__class__.__name__.lower()) and any(['lora' in n.lower() for n in class_type_names]):
            matrix = m.weight.data.clone()
            print(name, m.weight.data.device)
            # add lora layers to the original matrix
            if isinstance(m.lora_A, nn.ModuleDict):
                lora_A = m.lora_A['default'].weight.data.clone()
                lora_B = m.lora_B['default'].weight.data.clone()
            else:
                lora_A = m.lora_A.data.clone()
                lora_B = m.lora_B.data.clone()
            lora_matrix = (lora_B @ lora_A)
            if add_W:
                print("Calculating the ESD of lora matrix and original W......")
                matrix += lora_matrix
            else:
                print("Calculating the ESD of lora matrices only......")
                matrix = lora_matrix

            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            # print(matrix)
            matrix = matrix.to(torch.float32)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()

            if len(eigs) < eigs_num_thresh:
                continue

            if filter_zeros:
                sr_eigs = eigs[eigs > EVALS_THRESH]
                if len(sr_eigs) == 0:
                    sr_eigs = eigs
            else:
                sr_eigs = eigs

            if sr_mid_pos is not None:
                mid = int(len(sr_eigs) / sr_mid_pos)
                sr_eigs = sr_eigs[mid: ]
            eigs_sum = torch.sum(sr_eigs)
            max_eigs = torch.max(sr_eigs)
            stable_rank = eigs_sum / max_eigs
            norm_stable_rank = eigs_sum / len(sr_eigs)
            mid_eig = sr_eigs[len(sr_eigs) // 2]
            eig_ratio = max_eigs / mid_eig

            results['stable_rank'].append(stable_rank.item())
            results['norm_stable_rank'].append(norm_stable_rank.item())
            if init_stable_rank is not None:
                results['init_norm_stable_rank'].append(norm_stable_rank.item() / init_stable_rank[len(results['init_norm_stable_rank'])])
            else:
                results['init_norm_stable_rank'].append(0)
            results['eig_ratio'].append(eig_ratio.item())
            
            if filter_zeros:
                #print(f"{name} Filter Zero")
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    #print(f"{name} No non-zero eigs, use original total eigs")
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                #print(f"{name} Skip Filter Zero")
                nz_eigs = eigs
                N = len(nz_eigs)

            nz_eigs = nz_eigs.to("cuda:0")
            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n).cuda()
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()

            results['spectral_norm'].append(spectral_norm)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())
    
    return results

def net_cul_esd_estimator(
            net=None,
            ori_net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5, 
            filter_zeros=False,
            init_stable_rank=None,
            sr_mid_pos=None):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'stable_rank':[],
        'norm_stable_rank':[],
        'init_norm_stable_rank':[],
        'eig_ratio':[],
        }
    print("=================================")
    print(f"anslyzing the cumulated weight matrices: fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for (name, m), (ori_name, ori_m) in zip(net.named_modules(), ori_net.named_modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone()
            ori_matrix = ori_m.weight.data.clone()
            if any(dim < 10 for dim in matrix.shape):
                break
            # i have checked that the multiplication won't affect the weights value
            #print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
                ori_matrix = torch.flatten(ori_matrix, start_dim=2) * math.sqrt(conv_norm)
                ori_matrix = ori_matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            eigs = torch.square(torch.linalg.svdvals(matrix - ori_matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()

            if filter_zeros:
                sr_eigs = eigs[eigs > EVALS_THRESH]
                if len(sr_eigs) == 0:
                    sr_eigs = eigs
            else:
                sr_eigs = eigs

            if sr_mid_pos is not None:
                mid = int(len(sr_eigs) / sr_mid_pos)
                sr_eigs = sr_eigs[mid: ]

            eigs_sum = torch.sum(sr_eigs)
            max_eigs = torch.max(sr_eigs)
            stable_rank = eigs_sum / max_eigs
            norm_stable_rank = eigs_sum / len(sr_eigs)
            mid_eig = sr_eigs[len(sr_eigs) // 2]
            eig_ratio = max_eigs / mid_eig

            results['stable_rank'].append(stable_rank.item())
            results['norm_stable_rank'].append(norm_stable_rank.item())
            if init_stable_rank is not None:
                results['init_norm_stable_rank'].append(norm_stable_rank.item() / init_stable_rank[len(results['init_norm_stable_rank'])])
            else:
                results['init_norm_stable_rank'].append(0)
            results['eig_ratio'].append(eig_ratio.item())
            
            if filter_zeros:
                #print(f"{name} Filter Zero")
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    #print(f"{name} No non-zero eigs, use original total eigs")
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                #print(f"{name} Skip Filter Zero")
                nz_eigs = eigs
                N = len(nz_eigs)

            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n).cuda()
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()

            results['spectral_norm'].append(spectral_norm)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())
    
    return results




def get_layer_temps(args, temp_balance, n_alphas, epoch_val, metrics=None, hyperparam='lr'):
    """

    Args:
        temp_balance (_type_): method type 
        n_alphas (_type_): all the metric values
        epoch_val (_type_): basic untuned learning rate
        metrics (_type_): mainly to obtain layer names
    """
    n = len(n_alphas)
    idx = [i for i in range(n)]
    temps = np.array([epoch_val] * n)

    if temp_balance == 'tbr':
        print("--------------------> Use tbr method to schedule")
        idx = np.argsort(n_alphas)
        #temps = [2 * epoch_val * (0.35 + 0.15 * 2 * i / n) for i in range(n)]
        temps = [epoch_val * (args.lr_min_ratio + args.lr_slope * i / n) for i in range(n)]
        #print("temps",    args.lr_min_ratio,  args.lr_slope )
        #print("temps", temps)
        # Examples:
        # 4 3 5 -> argsort -> 1 0 2
        # temps = [0.7, 1, 1.3]
        # zip([1, 0, 2], [0.7, 1, 1.3]) -> [(1, 0.7), (0, 1), (2, 1.3)] -> [(0, 1),(1, 0.7),(2, 1.3)]
        return [value for _, value in sorted(list(zip(idx, temps)), key=itemgetter(0))]
    elif temp_balance == 'tb_linear_map':
        #print("!!!!!!!!!!", epoch_val, args.lr_min_ratio, args.lr_min_ratio + args.lr_slope)
        lr_range = [args.lr_min_ratio * epoch_val,  (args.lr_min_ratio + args.lr_slope) * epoch_val]
        score_range = [min(n_alphas),  max(n_alphas)]
        temps = np.interp(n_alphas, score_range, lr_range)
        #print(temps)
        if args.tb_lr_normalize == 'True':
            temps = [lr / np.mean(temps) * epoch_val for lr in temps]
            return temps
        return temps
    
    elif temp_balance == 'tb_sqrt':
        temps = np.sqrt(n_alphas)/np.sum(np.sqrt(n_alphas)) * n * epoch_val
        return temps
    
    elif temp_balance == 'tb_log2':
        temps = np.log2(n_alphas)/np.sum(np.log2(n_alphas)) * n * epoch_val
        return temps

    elif temp_balance == 'tb_std_sigmoid':
        alpha_mean = np.mean(n_alphas)
        diffs = n_alphas - alpha_mean
        def stage_sigmoid(x):
            if x > 0:
                return args.sigmoid_max * (1 / (1 + np.exp(-1 * args.exp_temp  * x)) - 0.5)
            else:
                return args.sigmoid_min * (1 / (1 + np.exp(-1 * args.exp_temp  * x)) - 0.5)
            
        factors = np.array([stage_sigmoid(x) for x in diffs])
        print("diffs", diffs)
        print("factors", factors)
        scaling = 10 ** factors
        temps = epoch_val * scaling
        return temps
    
    elif temp_balance == 'tb_layer_groups':
        # assert metrics is not None and args.lr_scale != 1
        layer_names = metrics['longname']
        layer_dict = {}
        idx_list = []
        
        # parse and group layer names
        for idx, layer in enumerate(layer_names):
            # parse layer name to group different types of layers
            match = re.search(r'(.*?)\.(\d+)\.(.*)', layer)
            if match:
                idx_list.append(idx)
                before = match.group(1)
                index = int(match.group(2))
                after = match.group(3)
                type_name = before + '.' + after
                
                if type_name in layer_dict:
                    layer_dict[type_name].append(idx)
                else:
                    layer_dict[type_name] = []
                    layer_dict[type_name].append(idx)
                    
        new_val = np.zeros(len(layer_names))
        for layer_group, layer_idxs in layer_dict.items():
            group_metrics = [n_alphas[i] for i in layer_idxs]
            score_range = [min(group_metrics),  max(group_metrics)]
            lr_range = [args.lr_min_ratio * epoch_val,  (args.lr_min_ratio + args.lr_slope) * epoch_val]
            # group_new_val = calc_ranked_metrics(group_metrics, epoch_val, args.lr_scale)
            group_new_val = np.interp(group_metrics, score_range, lr_range[::-1])
            for idx, val in enumerate(layer_idxs):
                new_val[val] = group_new_val[idx]
        for idx in range(len(layer_names)):
            if idx not in idx_list:
                new_val[idx] = epoch_val
                
        return new_val
    
    elif temp_balance == 'tb_layer_blocks':
        # assert metrics is not None and args.lr_scale != 1
        layer_names = metrics['longname']
        block_dict = {}
        idx_list = []
        
        # parse and group layer names
        for idx, layer in enumerate(layer_names):
            # parse layer name to group different types of layers
            match = re.search(r'(.*?)\.(\d+)\.(.*)', layer)
            if match and (args.lora_esd == "False" or any(s in layer for s in args.lora_target_modules)):
                idx_list.append(idx)
                before = match.group(1)
                index = int(match.group(2))
                after = match.group(3)
                type_name = before + '.' + after
                
                if index in block_dict:
                    block_dict[index].append(idx)
                else:
                    block_dict[index] = []
                    block_dict[index].append(idx)

        block_mean_metric = {}
        block_param = {}
        for block_num, block_idxs in block_dict.items():
            block_mean_metric[block_num] = np.mean([n_alphas[i] for i in block_idxs])
        block_metric_list = [alpha for _, alpha in block_mean_metric.items()]
        if args.schedule_func == "linear_map":
            score_range = [min(block_metric_list),  max(block_metric_list)]
            if hyperparam == "lr":
                param_range = [args.lr_min_ratio * epoch_val,  (args.lr_min_ratio + args.lr_slope) * epoch_val]
            elif hyperparam == "wd":
                param_range = [args.wd_min_ratio * epoch_val,  (args.wd_min_ratio + args.wd_slope) * epoch_val]
            else:
                raise NotImplementedError
            print(param_range)
            # block_new_val = np.interp(block_metric_list, score_range, lr_range[::-1])
            block_new_val = np.interp(block_metric_list, score_range, param_range)
        elif args.schedule_func == "rank":
            idx = np.argsort(block_metric_list)
            if hyperparam == "lr":
                print(n, len(block_metric_list))
                temps = [epoch_val * (args.lr_min_ratio + args.lr_slope * i / len(block_metric_list)) for i in range(len(block_metric_list))]
                print("Temps**********************************")
                print(temps)
            elif hyperparam == "wd":
                temps = [epoch_val * (args.wd_min_ratio + args.wd_slope * i / len(block_metric_list)) for i in range(len(block_metric_list))]
            
            block_new_val = [value for _, value in sorted(list(zip(idx, temps)), key=itemgetter(0))]

        elif args.schedule_func == "sigmoid":
            alpha_mean = np.mean(block_metric_list)
            diffs = block_metric_list - alpha_mean
            def stage_sigmoid(x):
                if x > 0:
                    return args.sigmoid_max * (1 / (1 + np.exp(-1 * args.exp_temp  * x)) - 0.5)
                else:
                    return args.sigmoid_min * (1 / (1 + np.exp(-1 * args.exp_temp  * x)) - 0.5)
                
            factors = np.array([stage_sigmoid(x) for x in diffs])
            print("diffs", diffs)
            print("factors", factors)
            scaling = 10 ** factors
            block_new_val = epoch_val * scaling
        
        for i, idx in enumerate(block_mean_metric.keys()):
            block_param[idx] = block_new_val[i]

        if args.tb_unit == "block":
            new_val = np.zeros(len(layer_names))
            for block_num, block_idxs in block_dict.items():
                for idx, val in enumerate(block_idxs):
                    new_val[val] = block_param[block_num]
            for idx in range(len(layer_names)):
                if idx not in idx_list:
                    new_val[idx] = epoch_val
            print(new_val)
            if args.tb_lr_normalize == 'True':
                new_val = [lr / np.mean(new_val) * epoch_val for lr in new_val]
            print(new_val)
            return new_val
        
        elif args.tb_unit == "layer":
            new_val = np.zeros(len(layer_names))
            for block_num, block_idxs in block_dict.items():
                block_metrics = [n_alphas[i] for i in block_idxs]
                score_range = [min(block_metrics),  max(block_metrics)]
                if hyperparam == "lr":
                    param_range = [args.lr_min_ratio * block_param[block_num],  (args.lr_min_ratio + args.lr_slope) * block_param[block_num]]
                elif hyperparam == "wd":
                    param_range = [args.wd_min_ratio * block_param[block_num],  (args.wd_min_ratio + args.wd_slope) * block_param[block_num]]
                else:
                    raise NotImplementedError
                block_new_val = np.interp(block_metrics, score_range, param_range)
                for idx, val in enumerate(block_idxs):
                    new_val[val] = block_new_val[idx]
            for idx in range(len(layer_names)):
                if idx not in idx_list:
                    new_val[idx] = epoch_val

            if args.tb_lr_normalize == 'True':
                new_val = [lr / np.mean(new_val) * epoch_val for lr in new_val]

            return new_val
        
        elif args.tb_unit == "sliding_window" and args.tb_window_idx >= 0:
            assert args.tb_window_scale is not None, "Must select a valid scale."
            assert args.tb_window_idx < len(block_dict), "Window must within valid range."
            new_val = np.zeros(len(layer_names))
            for block_num, block_idxs in block_dict.items():
                if block_num == args.tb_window_idx:
                    for idx, val in enumerate(block_idxs):
                        new_val[val] = args.tb_window_scale * epoch_val
                else:
                    for idx, val in enumerate(block_idxs):
                        new_val[val] = epoch_val
            for idx in range(len(layer_names)):
                if idx not in idx_list:
                    new_val[idx] = epoch_val

            return new_val

        else:
            raise NotImplementedError
         
    else:
        raise NotImplementedError

