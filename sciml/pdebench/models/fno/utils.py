# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     utils.py
  Authors:  Timothy Praditia (timothy.praditia@iws.uni-stuttgart.de)
            Raphael Leiteritz (raphael.leiteritz@ipvs.uni-stuttgart.de)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Francesco Alesiani (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
from .fno import SpectralConv2d_fast


class FNODatasetSingle(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 total_num_train_samples=10000000000,
                 num_samples_max=-1
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        print('total_num_train_samples',total_num_train_samples)
        ###often 1000
        test_data_num = int(total_num_train_samples / 9)  
        train_data_num = int(total_num_train_samples/reduced_batch)
        total_data_num =  train_data_num + test_data_num


        # Define path to files
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        if filename[-2:] != 'h5':
            print(f".HDF5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        print("1D data is assumed hereafter")
                        self.data = np.zeros([total_data_num,   #idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                            dtype=np.float32)
                        #density
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,0] = _data   # batch, x, t, ch

                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,1] = _data   # batch, x, t, ch

                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)
                        
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(self.data.shape)
                    if len(idx_cfd)==4:  # 2D
                        print("2D data is assumed hereafter")
                        self.data = np.zeros([total_data_num,               #idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,0] = _data   # batch, x, t, ch

                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,1] = _data   # batch, x, t, ch

                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                
                    if len(idx_cfd)==5:  # 3D
                        print("2D data is assumed hereafter")
                        self.data = np.zeros([total_data_num,           #idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,0] = _data   # batch, x, t, ch

                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,1] = _data   # batch, x, t, ch

                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,3] = _data   # batch, x, t, ch

                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                        self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                    ::reduced_resolution,\
                                                                    ::reduced_resolution]
                                                                    
                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        print("Scalar equations, 1D data is assumed hereafter")
                        _data_sub1 = _data[:test_data_num,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,::reduced_resolution_t,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data = _data[:, :, :, None]  # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        print("Scalar equations, 2D data is assumed hereafter")
                        _data_sub1 = _data[:test_data_num,:,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:,:,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data_sub1 = _data[:test_data_num, None,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data[test_data_num:, None,::reduced_resolution,::reduced_resolution]
                        _data_sub2 = _data_sub2[::reduced_batch,:,:,:]
                        _data = np.concatenate((_data_sub1, _data_sub2), axis=0)

                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        elif filename[-2:] == 'h5':  # SWE-2D (RDB)
            print(f".H5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]

                _data_sub1 = _data[:test_data_num, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data_sub2 = _data[test_data_num:, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data_sub2 = _data_sub2[::reduced_batch, :, : ,: ...]
                _data = np.concatenate((_data_sub1, _data_sub2), axis=0)
                
                _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data
                self.grid = _grid
                self.tsteps_t = tsteps_t

        if num_samples_max>0:
            num_samples_max  = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        #test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_data_num]
        else:
            self.data = self.data[test_data_num:total_data_num]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid



import torch
import torch.nn as nn
from operator import itemgetter
import numpy as np
import math


def net_esd_estimator(
            net=None,
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
        'eig_number_filtered':[],
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
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

            if filter_zeros:
                sr_eigs = eigs[eigs > EVALS_THRESH]
                eig_number_f = (len(sr_eigs))
                if len(sr_eigs) == 0:
                    sr_eigs = eigs
            else:
                eig_number_f = (len(eigs))
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
            results['eig_number_filtered'].append(eig_number_f)
    
    return results



def get_layer_temps(lr_min_ratio, lr_slope, temp_balance, n_alphas, epoch_val, exp_temp):
    """

    Args:
        temp_balance (_type_): method type 
        n_alphas (_type_): all the metric values
        epoch_val (_type_): basic untuned learning rate
    """
    n = len(n_alphas)
    idx = [i for i in range(n)]
    temps = np.array([epoch_val] * n)

    if temp_balance == 'tb_linear_map':
        lr_range = [lr_min_ratio * epoch_val,  (lr_min_ratio + lr_slope) * epoch_val]
        score_range = [min(n_alphas),  max(n_alphas)]
        temps = np.interp(n_alphas, score_range, lr_range)
        return temps

    elif temp_balance == 'tb_std_sigmoid':
        sigmoid_min = lr_min_ratio
        sigmoid_max = lr_slope
        
        alpha_mean = np.mean(n_alphas)
        diffs = n_alphas - alpha_mean
        def stage_sigmoid(x):
            if x > 0:
                return sigmoid_max * (1 / (1 + np.exp(-1 * exp_temp  * x)) - 0.5)
            else:
                return sigmoid_min * (1 / (1 + np.exp(-1 * exp_temp  * x)) - 0.5)
            
        factors = np.array([stage_sigmoid(x) for x in diffs])
        #print("diffs", diffs)
        #print("factors", factors)
        
        scaling = 10 ** factors
        temps = epoch_val * scaling
        return temps 
    

    else:
        raise NotImplementedError






