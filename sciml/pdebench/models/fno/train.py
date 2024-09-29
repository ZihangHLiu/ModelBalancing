from math import log
import sys
from matplotlib.path import Path
import pandas as pd
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import os
import operator
from functools import reduce
from functools import partial
import csv
from timeit import default_timer


# torch.manual_seed(0)
# np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .fno import FNO1d, FNO2d, FNO3d
from .utils import FNODatasetSingle, get_layer_temps, net_esd_estimator
from .metrics import metrics
from .optimizer import Adam

def run_training(if_training,
                 continue_training,
                 num_workers,
                 modes,
                 width,
                 initial_step,
                 t_train,
                 num_channels,
                 batch_size,
                 epochs,
                 learning_rate,
                 scheduler_step,
                 scheduler_gamma,
                 model_update,
                 flnm,
                 single_file,
                 reduced_resolution,
                 reduced_resolution_t,
                 reduced_batch,
                 plot,
                 channel_plot,
                 x_min,
                 x_max,
                 y_min,
                 y_max,
                 t_min,
                 t_max,
                 ############TB#############
                 temp_balance_lr,
                 fix_fingers,
                 lr_min_ratio,
                 lr_slope,
                 tb_interval_ep,
                 eigs_thres,
                 tb_batchnorm,
                 tb_metric,
                 training_start_time,
                 weight_decay,
                 tb_hyperparam,
                 exp_temp, 
                 ############TB##############
                 base_path='../data/',
                 training_type='autoregressive',
                 train_log_path = None,
                 checkpoint_path = None,
                 ):
    #####log_path checking
    assert train_log_path, checkpoint_path is not None
    
    print(f'Epochs = {epochs}, learning rat = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}')
    print('TIME M ON.')

    ################################################################
    # load data
    ################################################################
    print(flnm)
    if 'Darcy' in flnm or "1D_CFD" in flnm:
        total_num_train_samples = 9000
    elif '2D_CFD' in flnm:
        total_num_train_samples = 9000

    if single_file:
        # filename
        model_name = flnm[:-5] + '_FNO'
        print("FNODatasetSingle")
        print(base_path)
        # Initialize the dataset and dataloader
        train_data = FNODatasetSingle(flnm,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder = base_path,
                                total_num_train_samples = total_num_train_samples,
                                )
        val_data = FNODatasetSingle(flnm,
                              reduced_resolution=reduced_resolution,
                              reduced_resolution_t=reduced_resolution_t,
                              reduced_batch=reduced_batch,
                              initial_step=initial_step,
                              if_test=True,
                              saved_folder = base_path,
                              total_num_train_samples = total_num_train_samples,
                              )

    print(len(train_data))
    print(len(val_data))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)

    if "Darcy" or "1D_CFD"in flnm:
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size*10,
                                                num_workers=num_workers, shuffle=False)
    else:
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                num_workers=num_workers, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
    if dimensions == 4:
        model = FNO1d(num_channels=num_channels,
                      width=width,
                      modes=modes,
                      initial_step=initial_step).to(device)
    elif dimensions == 5:
        model = FNO2d(num_channels=num_channels,
                      width=width,
                      modes1=modes,
                      modes2=modes,
                      initial_step=initial_step).to(device)
    elif dimensions == 6:
        model = FNO3d(num_channels=num_channels,
                      width=width,
                      modes1=modes,
                      modes2=modes,
                      modes3=modes,
                      initial_step=initial_step).to(device)
        
    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]

    if if_training:
        model_path = model_name + training_start_time + ".pt"
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    print('TIME M ON.')
    ########################################TB Code#############################################
    model.to(device)
    model.eval()
    
    dir = os.path.join(train_log_path, 'stats')
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    dir = os.path.join(train_log_path, 'lr_stats')
    if not os.path.exists(dir):
        os.makedirs(dir)

    ####Need to fix the bug
    tb_metrics = net_esd_estimator(model, 
                        EVALS_THRESH = 0.00001,
                        bins = 100,
                        fix_fingers=fix_fingers,
                        xmin_pos=2,
                        filter_zeros=True)
    init_stable_rank = tb_metrics['stable_rank']
    assert init_stable_rank is not None, "init stable rank should not be None"
   
    pd.DataFrame(tb_metrics).to_csv(os.path.join(train_log_path, 'stats',  f"tb_metrics.csv")) 
    
    metric_summary = {}
    for key in tb_metrics:
        if key != 'eigs' and key != 'longname':
            metric_summary[key] = np.mean(tb_metrics[key])

    #######################  Filter out layers who has little amount of eigenvalues ##########################
    layer_with_few_eigs = []
    for i, name in enumerate(tb_metrics['longname']):
        print(f"layer [{name}] has {tb_metrics['eig_number_filtered'][i]} eigenvalues")
        if tb_metrics['eig_number_filtered'][i] <= eigs_thres:
            layer_with_few_eigs.append(name)


    layer_stats=pd.DataFrame({key:tb_metrics[key] for key in tb_metrics if key!='eigs'})      
    layer_stats_origin = layer_stats.copy()
    
    pd.DataFrame(layer_with_few_eigs).to_csv(os.path.join(train_log_path, 'stats',  f"removed layers.csv")) 
    layer_stats_origin.to_csv(os.path.join(train_log_path, 'stats',  f"origin_layer_stats_epoch_start.csv"))
    np.save(os.path.join(train_log_path, 'stats', 'esd_epoch_0.npy'), tb_metrics)


    ###################End  ESD analysis############################
    ##################################################################
    ######################  TBR scheduling ##########################
    ##################################################################
    if temp_balance_lr != 'None':
        print("--------------Enable temp balance --------------")
        

        ####remove with the few eig values
        drop_layers = layer_stats['longname'].isin(layer_with_few_eigs)
        layer_stats = layer_stats[~drop_layers]
        
        metric_scores = np.array(layer_stats[tb_metric])
        #args, temp_balance, n_alphas, epoch_val
        if tb_hyperparam == 'lr':
            scheduled_hyperparam = get_layer_temps(lr_min_ratio, lr_slope, temp_balance=temp_balance_lr, n_alphas=metric_scores, epoch_val=learning_rate, exp_temp=exp_temp)
            layer_stats['scheduled_lr'] = scheduled_hyperparam
        else:
            raise ValueError("tb_hyperparam should be either lr")

        # these params should be tuned
        layer_name_to_tune = list(layer_stats['longname'])
        all_params = []
        params_to_tune_ids = []

        # these params should be tuned
        for name, module in model.named_modules():
            # these are the conv layers analyzed by the weightwatcher
            print(name)
            if name in layer_name_to_tune:
                params_to_tune_ids += list(map(id, module.parameters()))
                
                if tb_hyperparam == 'lr':
                    scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
                    all_params.append({'params': module.parameters(), 'lr': scheduled_lr})        
                else:
                    raise ValueError("tb_hyperparam should be either lr")


            # decide should we tune the batch norm accordingly,  is this layer batchnorm and does its corresponding conv in layer_name_to_tune
            elif tb_batchnorm == True \
                    and isinstance(module, nn.BatchNorm2d) \
                        and name.replace('bn', 'conv') in layer_name_to_tune:
                params_to_tune_ids += list(map(id, module.parameters()))

                if tb_hyperparam == 'lr':
                    scheduled_lr = layer_stats[layer_stats['longname'] == name.replace('bn', 'conv')]['scheduled_lr'].item()
                    all_params.append({'params': module.parameters(), 'lr': scheduled_lr})
                else:
                    raise ValueError("tb_hyperparam should be either lr")
            # another way is to add a else here and append params with lr0

        # those params are untuned
        untuned_params = filter(lambda p: id(p) not in params_to_tune_ids, model.parameters())
        # if tb_hyperparam is lr, then we need to tune the learning rate
        #######maybe could be a bug here
        if tb_hyperparam == 'lr':
            all_params.append({'params': untuned_params,  'lr': learning_rate}) 
        else:
            raise ValueError("tb_hyperparam should be either lr")

        
        optimizer = Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
    else: 
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ##################################################################TB END#############################
            
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0
    print(if_training)
    
    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1., 1., 1.
        errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                       model_name, x_min, x_max, y_min, y_max,
                       t_min, t_max, initial_step=initial_step)
        print(errs)
        pickle.dump(errs, open(model_name+'.pickle', "wb"))
        
        return
    

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        print('Restoring model (that is the network\'s weights) from file...')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        
        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        start_epoch = checkpoint['epoch']
        loss_val_min = checkpoint['loss']
    
    for ep in range(start_epoch, epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        train_iter_lenth = 0
        
        for xx, yy, grid in train_loader:
            loss = 0
            
            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)

            # Initialize the prediction tensor
            pred = yy[..., :initial_step, :]
            # Extract shape of the input tensor for reshaping (i.e. stacking the
            # time and channels dimension together)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)
    
            if training_type in ['autoregressive']:
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)
                    
                    # Extract target at current time step
                    y = yy[..., t:t+1, :]

                    # Model run
                    im = model(inp, grid)

                    # Loss calculation
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
        
                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)
        
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)


                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()
                ####train_iteration
                train_iter_lenth += 1
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if training_type in ['single']:
                x = xx[..., 0 , :]
                y = yy[..., t_train-1:t_train, :]
                pred = model(x, grid)
                _batch = yy.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
    
                train_l2_step += loss.item()
                train_l2_full += loss.item()

                train_iter_lenth += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print('train_loss_lenth', train_iter_lenth)

        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            model.eval()
            test_iter_lenth = 0

            with torch.no_grad():
                for xx, yy, grid in val_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    
                    if training_type in ['autoregressive']:
                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)
                
                        for t in range(initial_step, yy.shape[-2]):
                            inp = xx.reshape(inp_shape)
                            y = yy[..., t:t+1, :]
                            im = model(inp, grid)
                            _batch = im.size(0)
                            loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                            pred = torch.cat((pred, im), -2)
                
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
                        ####
                        test_iter_lenth += 1

                    if training_type in ['single']:
                        x = xx[..., 0 , :]
                        y = yy[..., t_train-1:t_train, :]
                        pred = model(x, grid)
                        _batch = yy.size(0)
                        loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                        test_iter_lenth += 1
                        val_l2_step += loss.item()
                        val_l2_full += loss.item()
                
                print('test_loss_lenth', test_iter_lenth)
                
                if  val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    model_save_path = os.path.join(checkpoint_path, model_path)
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_val_min
                        }, model_save_path)
                
        scheduler.step()
        ###############################TB Code############################################
        ###################################################################################
        #####scaling ratio change for TB 
        current_lr = optimizer.param_groups[-1]['lr']
        current_weight_decay = optimizer.param_groups[-1]['weight_decay']

        t_tb_start = default_timer()
        if (ep) % tb_interval_ep == 0 or ep == 1: 
            tb_metrics = net_esd_estimator(model, 
                            EVALS_THRESH = 0.00001,
                            bins = 100,
                            fix_fingers=fix_fingers,
                            xmin_pos=2,
                            filter_zeros=True,
                            init_stable_rank=init_stable_rank,
                            sr_mid_pos=2)

            for key in tb_metrics:
                if key != 'eigs' and key != 'longname':
                    metric_summary[key] = np.mean(tb_metrics[key])

            layer_with_few_eigs = []    
            #######################  Filter out layers who has little amount of eigenvalues ##########################
            for i, name in enumerate(tb_metrics['longname']):
                print(f"layer [{name}] has {tb_metrics['eig_number_filtered'][i]} eigenvalues")
                if tb_metrics['eig_number_filtered'][i] < eigs_thres:
                    layer_with_few_eigs.append(name)


            layer_stats=pd.DataFrame({key:tb_metrics[key] for key in tb_metrics if key!='eigs'})      
                
            # save metrics to disk and ESD
            layer_stats_origin = layer_stats.copy()
            #origin saving
            layer_stats_origin.to_csv(os.path.join(train_log_path, 'stats',  f"origin_layer_stats__epoch_{ep}.csv"))
            #.npy saving
            np.save(os.path.join(train_log_path, 'stats', f'esd__epoch_{ep}_{fix_fingers}.npy'), tb_metrics)
        else:
            pass

        ######################  TBR scheduling ##########################
        if temp_balance_lr != 'None':
            
            if ep < 10:
                print("---------- Schedule by Temp Balance---------------")
            
            assert len(layer_stats) > 0, "in TBR, every epoch should has an updated metric summary"

            ####remove with the few eig values
            drop_layers = layer_stats['longname'].isin(layer_with_few_eigs)
            layer_stats = layer_stats[~drop_layers]

            metric_scores = np.array(layer_stats[tb_metric])

            if tb_hyperparam == 'lr':
                scheduled_lr = get_layer_temps(lr_min_ratio, lr_slope, temp_balance=temp_balance_lr, n_alphas=metric_scores, epoch_val=current_lr, exp_temp=exp_temp)
                layer_stats['scheduled_lr'] = scheduled_lr
            else:
                raise ValueError("tb_hyperparam should be either lr")
            

            # these params should be tuned
            layer_name_to_tune = list(layer_stats['longname'])
            all_params_lr = []
            all_params_wd = []
            params_to_tune_ids = []
            c = 0
            
            #####check the few eig values layers were removed
            for name, module in model.named_modules():
                if name in layer_name_to_tune:
                    assert name not in layer_with_few_eigs


            for name, module in model.named_modules():
                if name in layer_name_to_tune:
                    params_to_tune_ids += list(map(id, module.parameters()))

                    if tb_hyperparam == 'lr':
                        scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
                        all_params_lr.append(scheduled_lr)
                        c = c + 1
                    else:
                        raise ValueError("tb_hyperparam should be either lr")
                    
                # decide should we tune the batch norm accordingly,  is this layer batchnorm and does its corresponding conv in layer_name_to_tune
                elif tb_batchnorm == True \
                    and isinstance(module, nn.BatchNorm2d) \
                        and name.replace('bn', 'conv') in layer_name_to_tune:
                    params_to_tune_ids += list(map(id, module.parameters()))

                    if tb_hyperparam == 'lr':
                        scheduled_lr = layer_stats[layer_stats['longname'] == name.replace('bn', 'conv')]['scheduled_lr'].item()
                        all_params_lr.append(scheduled_lr)
                        c = c + 1
                    else:
                        raise ValueError("tb_hyperparam should be either lr")
            
            if ep % tb_interval_ep == 0:
                layer_stats.to_csv(os.path.join(train_log_path, 'lr_stats', f"layer_stats_with_lr__epoch_{ep}.csv"))
            
            for index, param_group in enumerate(optimizer.param_groups):
                if index <= c - 1:
                    if tb_hyperparam == 'lr':
                        param_group['lr'] = all_params_lr[index]
                    else:
                        raise ValueError("tb_hyperparam should be either lr")
                else:

                    if tb_hyperparam == 'lr':
                        param_group['lr'] = current_lr
                    else:
                        raise ValueError("tb_hyperparam should be either lr")  
        #####################if TB is not used########################
        else:
            if ep < 10:
                print("------------>  Schedule by default")
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                param_group['weight_decay'] = current_weight_decay
        t_tb_end = default_timer()

        t2 = default_timer()
        # model evaluation
        model.eval()
        Lx, Ly, Lz = 1., 1., 1.
        errs = metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot,
                    model_save_path, x_min, x_max, y_min, y_max,
                    t_min, t_max, initial_step=initial_step)

        csv_file_path = os.path.join(train_log_path, model_name + '_errs.csv')
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['RMSE', 'nRMSE', 'RMSE of conserved variables', 'Maximum value of rms error', 'RMSE at boundaries', 'RMSE in Fourier space'])
            writer.writerow([errs[0], errs[1], errs[2], errs[3], errs[4], errs[5]])
        
        # Check if the file exists, if not, create the file and write the header
        
        csv_file_path = os.path.join(train_log_path, model_name + '.csv')
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Epoch', 'loss', 't2-t1', 'trainL2', 'testL2', 't_tb'])
            writer.writerow([ep, loss.item(), t2 - t1, train_l2_full, val_l2_full, t_tb_end-t_tb_start])
        
        print('epoch: {0}, loss: {1:.5f}, t2-t1: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}, t_tb: {5:.5f}'\
                .format( ep, loss.item(), t2 - t1, train_l2_full, val_l2_full, (t_tb_end-t_tb_start) ))
        
    

        
            
if __name__ == "__main__":
    
    run_training()
    print("Done.")

