import os
import pickle
import math
import random
import numpy as np
from scipy.special import lambertw
import scipy.stats as st
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from util.create_dataset import MyDataset


#fxn to preprocess ColdHardiness dataset, get into common form that can be used by run_sampling and run_weighting_eval
def run_preprocess_data_CH(args):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    #model set up
    model = args.nn_model(feature_len, args.n_experts)
    model.to(args.device)
    #load expert model
    check_path = os.path.join('./models', args.dataset_name, "embed_train_lr4", args.current_task_name, ("trial_" + str(args.trial)), args.experiment+'_setting_'+"leaveoneout"+".pt")
    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    #use a different set of experts
    if(args.other_experts is not None):
        with open('./models/'+args.dataset_name+'/extra_embeds/'+"trial_" + str(args.trial)+'_uniform_'+args.other_experts+'.pkl','rb') as f:
            extra_weights = pickle.load(f)
        #replace model embeddings with new experts
        extra_tasks, _ = extra_weights[args.current_task_name].shape        
        concat_weights = (extra_weights[args.current_task_name]).to(args.device)
        model.embedding = nn.Embedding.from_pretrained(concat_weights)
        args.n_experts = extra_tasks
    
    #set up eval criterion
    model.to(args.device)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    
    with torch.no_grad():
        model.eval()
        
        #get training data from dataset form
        train_dataset = MyDataset(dataset['train'])
        (x, y, _) = train_dataset[:]
        x_torch = x.to(args.device)
        y_torch = y.to(args.device)
        #create structures to save data
        n_seasons, n_days, n_labels = y_torch.shape
        train_preds = np.zeros((n_seasons, n_days, n_labels, args.n_experts))
        train_actual = y.detach().cpu().numpy()
        train_losses = np.zeros((n_seasons, n_days, n_labels, args.n_experts))
        #get model predictions
        for t_id in range(args.n_experts):
            task_id_torch = (torch.ones((x.shape[0], x.shape[1]))*t_id).type(torch.LongTensor).to(args.device)
            out_primary, _, _ = model(x_torch, task_label=task_id_torch)
            train_preds[:,:,:,t_id] = torch.unsqueeze(out_primary[:,:,1],2).detach().cpu().numpy()
            train_losses[:,:,:,t_id] = criterion(torch.unsqueeze(out_primary[:,:,1],2), y_torch).detach().cpu().numpy()
    
        #get testing data from dataset form
        test_dataset = MyDataset(dataset['test'])
        (x, y, _) = test_dataset[:]
        x_torch = x.to(args.device)
        #create structures to save data
        n_seasons, n_days, n_labels = x_torch.shape
        test_preds = np.zeros((n_seasons, n_days, n_labels, args.n_experts))
        test_actual = y.detach().cpu().numpy()
        #get model predictions
        for t_id in range(args.n_experts):
            task_id_torch = (torch.ones((x.shape[0], x.shape[1]))*t_id).type(torch.LongTensor).to(args.device)
            out_primary, _, _ = model(x_torch, task_label=task_id_torch)
            test_preds[:,:,:,t_id] = torch.unsqueeze(out_primary[:,:,1],2).detach().cpu().numpy()
    
    #return preprocessed dataset
    return {"train_preds":train_preds,"train_actual":train_actual,"train_losses":train_losses,"test_preds":test_preds,"test_actual":test_actual}


#fxn to preprocess CropSim dataset, get into common form that can be used by run_sampling and run_weighting_eval
def run_preprocess_data_CropSim(args):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    criterion = nn.MSELoss(reduction='none')
    
    #get eval training data from dataset form
    train_dataset = MyDataset(dataset['train'])
    (x, y_act, _) = train_dataset[:]
    #set up expert training data
    train_dataset_expert = MyDataset(args.dataset_expert['train'])
    total_seasons = x.shape[0]
    train_batch_size = int(total_seasons/args.n_tasks)
    trainLoader_expert = DataLoader(train_dataset_expert, batch_size=train_batch_size, shuffle=False)
    #create structures to save data
    n_seasons, n_days, n_labels = y_act.shape
    train_preds = np.zeros((n_seasons, n_days, n_labels, args.n_experts))
    train_losses = np.zeros((n_seasons, n_days, n_labels, args.n_experts))
    #get expert predictions
    for t_id, (_, y_pred, _) in enumerate(trainLoader_expert):
        for sea in range(n_seasons):
            for label in range(n_labels):
                train_losses[sea,:,label,t_id] = (criterion(y_pred[sea,:,label], y_act[sea,:,label])).detach().cpu().numpy()
        train_preds[:,:,:,t_id] = y_pred.detach().cpu().numpy()
    train_actual = y_act.detach().cpu().numpy()
    
    #get eval test data from dataset form
    test_dataset = MyDataset(dataset['test'])
    (x_test, y_test_act, _) = test_dataset[:]
    #set up expert testing data
    test_dataset_expert = MyDataset(args.dataset_expert['test'])
    total_seasons = x_test.shape[0]
    test_batch_size = int(total_seasons/args.n_tasks)
    testLoader_expert = DataLoader(test_dataset_expert, batch_size=test_batch_size, shuffle=False)
    #create structures to save data
    n_seasons, n_days, n_labels = y_test_act.shape
    test_preds = np.zeros((n_seasons, n_days, n_labels, args.n_experts))
    test_actual = y_test_act.detach().cpu().numpy()
    #get expert predictions
    for t_id, (_, y_pred, _) in enumerate(testLoader_expert):
        test_preds[:,:,:,t_id] = y_pred.detach().cpu().numpy()
    
    #return preprocessed dataset
    return {"train_preds":train_preds,"train_actual":train_actual,"train_losses":train_losses,"test_preds":test_preds,"test_actual":test_actual}
    

#fxn to get samples and model weighting
def run_sampling(args, data_dict):
    #create structures to save data
    n_seasons, n_days, n_labels, n_experts = data_dict["train_preds"].shape
    sample_days = np.zeros((n_seasons, args.n_samples, n_labels))
    variances = np.zeros((n_seasons, n_days, n_labels))
    weights = np.zeros((n_seasons, n_labels, n_experts))
    
    for label in range(n_labels):
        for sea in range(n_seasons):
            #format data for sampling fxns
            actual_labels = np.squeeze(data_dict["train_actual"][sea,:,label])
            pred_labels = np.squeeze(data_dict["train_preds"][sea,:,label,:])
            expert_losses = np.squeeze(data_dict["train_losses"][sea,:,label,:])
            pred_history = np.squeeze(np.delete(data_dict["train_preds"][:,:,label,:], sea, axis=0))
            
            #get samples
            sd, var = args.policy_fxn(args, actual_labels, pred_labels, expert_losses, pred_history)
            sample_days[sea,:,label] = sd
            variances[sea,:,label] = var
    
            #get weights
            if args.policy == "baseline":
                samples_losses = np.ones(n_experts)
            else:
                samples_losses = np.sqrt(np.mean(expert_losses[sd,:], axis=0))
            wei = [math.exp(-1*args.a_weight*w) for w in samples_losses]
            total = sum(wei)
            wei = [w/total for w in wei]
            weights[sea,label,:] = wei
    
    #return sampling dataset
    return {"sample_days":sample_days,"variances":variances,"weights":weights}
    

#fxn to evaluate model weighting  
def run_weighting_eval(args, data_dict, sampling_dict):
    #create structures to save data
    n_test_seasons, n_days, n_labels = data_dict["test_actual"].shape
    n_train_seasons, _, _ = data_dict["train_actual"].shape
    weighted_preds = np.zeros((n_train_seasons, n_test_seasons, n_days, n_labels))
    #get weighted preds
    for train_sea in range(n_train_seasons):
        for label in range(n_labels):
            for test_sea in range(n_test_seasons):
                weighted_preds[train_sea,test_sea,:,label] = np.average(data_dict["test_preds"][test_sea,:,label,:], weights = sampling_dict["weights"][train_sea,label,:], axis=1)
    
    #create structures to save data
    losses = np.zeros((n_train_seasons, n_labels))
    nan_locs_actual = np.isnan(data_dict["test_actual"])
    criterion = nn.MSELoss(reduction='none')
    #get loss on weighted preds
    for train_sea in range(n_train_seasons):
        nan_locs_preds = np.isnan(weighted_preds[train_sea])
        for label in range(n_labels):
            nan_locs_combined = torch.from_numpy(np.logical_or(nan_locs_actual[:,:,label],nan_locs_preds[:,:,label]))
            losses[train_sea,label] = np.sqrt((criterion(torch.from_numpy(weighted_preds[train_sea,:,:,label]), torch.from_numpy(data_dict["test_actual"][:,:,label]))[~nan_locs_combined]).mean().item())
    
    #return evaluation dataset
    return {"weighted_preds":weighted_preds,"losses":losses}


#EMPIRICAL POLICIES VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
def run_empirical_1threshold_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #Empirical 1-threshold: use the past data to optimize a single threshold to achieve the best performance:
    #   Search through a range of reasonable thresholds and for each one compute a performance measure based on the historical data 
    #   using the current expert weights. Stop at the first value that exceeds the threshold (no waiting period). 
    #   For performance metric, compute the distribution/histogram of returns over the historical data. 
    #   Can use different statistics as measures (expected value or median)
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []    
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k    
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]

        #get metric value for all historical seasons and days
        n_sea_hist, _, _ = pred_history.shape
        sea_vars = np.zeros((n_sea_hist, len(substream)))
        for sea in range(n_sea_hist):
            for d, day in enumerate(substream):
                sea_vars[sea, d] = get_metric(pred_history[sea,day,:], weights, args.policy_metric)
        
        #find threshold with best performance over historical data
        thresholds = np.linspace(0, np.max(sea_vars), num=20)
        sea_thresholds = np.zeros((n_sea_hist,len(thresholds)))
        for s in range(n_sea_hist):
            for t, thresh in enumerate(thresholds):
                #gets first element above threshold, if none are above it takes the last
                sea_thresholds[s,t] = next((x for x in sea_vars[s,:] if x > thresh), sea_vars[s,-1]) 
        
        #pick which metric to select using
        if args.empirical_metric in ['median']:
            metric_vals = np.median(sea_thresholds, axis=0) #get median for each threshold
        if args.empirical_metric in ['mean']:
            metric_vals = np.mean(sea_thresholds, axis=0) #get mean for each threshold
        
        #get threshold with the max metric value
        threshold = thresholds[np.argmax(metric_vals)]
        
        #iterate over all days in substream
        sample = -1
        for day in substream:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>threshold and sample == -1:
                sample = day
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1: 
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances 


#PROPHET POLICIES VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
def run_prophet_median_policy(args, actual_labels, pred_labels, expert_losses, pred_history): 
    #Median:   use a constant threshold equal to the median of the max of each episode 
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #get predicted best historically for whole window
        n_sea_hist, _, _ = pred_history.shape
        sea_maxes = np.zeros((n_sea_hist, len(substream)))
        for sea in range(n_sea_hist):
            for d, day in enumerate(substream):
                sea_maxes[sea, d] = get_metric(pred_history[sea,day,:], weights, args.policy_metric)
        threshold = np.median(np.max(sea_maxes, axis=1))
        
        #iterate over all days in substream
        sample = -1
        for day in substream:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>threshold and sample == -1:
                sample = day
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances

def run_prophet_emax_policy(args, actual_labels, pred_labels, expert_losses, pred_history): 
    #EMax/2: This is the expectation of the maximums divided by 2
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #get predicted best historically for whole window
        n_sea_hist, _, _ = pred_history.shape
        sea_maxes = np.zeros((n_sea_hist, len(substream)))
        for sea in range(n_sea_hist):
            for d, day in enumerate(substream):
                sea_maxes[sea, d] = get_metric(pred_history[sea,day,:], weights, args.policy_metric)
        threshold = np.average(np.max(sea_maxes, axis=1))/2
        
        #iterate over all days in substream
        sample = -1
        for day in substream:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>threshold and sample == -1:
                sample = day
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances

def run_prophet_2threshold_policy(args, actual_labels, pred_labels, expert_losses, pred_history): 
    #2-thresholds:  For the first half use a threshold of (5/9)EMax and for the second half use (1/3)Emax
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #get predicted best historically for whole window
        n_sea_hist, _, _ = pred_history.shape
        sea_maxes = np.zeros((n_sea_hist, len(substream)))
        for sea in range(n_sea_hist):
            for d, day in enumerate(substream):
                sea_maxes[sea, d] = get_metric(pred_history[sea,day,:], weights, args.policy_metric)
        EMax = np.average(np.max(sea_maxes, axis=1))
        
        #iterate over all days in substream, split in half
        sample = -1
        split = len(substream)//2
        #For the first half use a threshold of (5/9)EMax
        threshold = (5/9)*EMax
        for day in substream[0:split]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>threshold and sample == -1:
                sample = day
    
        #for the second half use (1/3)Emax
        threshold = (1/3)*EMax
        for day in substream[split:]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>threshold and sample == -1:
                sample = day
        
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances

def run_prophet_nthreshold_policy(args, actual_labels, pred_labels, expert_losses, pred_history): 
	#n-threshold: (n is the length of your episode) At step i from 1 to n the threshold is 
    #alpha(i)*EMax for alpha(t) = 1-e^((i-n)/n)
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #get predicted best historically for whole window
        sea_maxes = []
        n_sea_hist, _, _ = pred_history.shape
        sea_maxes = np.zeros((n_sea_hist, len(substream)))
        for sea in range(n_sea_hist):
            for d, day in enumerate(substream):
                sea_maxes[sea, d] = get_metric(pred_history[sea,day,:], weights, args.policy_metric)
        EMax = np.average(np.max(sea_maxes, axis=1))
        
        #iterate over all days in substream
        sample = -1
        n = len(substream)
        for j, day in enumerate(substream):
            threshold = EMax * (1 - math.exp((j-n)/n))
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>threshold and sample == -1:
                sample = day
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances
    
def run_maxoracle_prophet_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #always pick the highest value sample
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #iterate over all days in substream
        sample = -1
        max_var = 0
        for day in substream: 
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take sample with greatest value
            if var>max_var:
                sample = day
                max_var = var
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances    
    

#SECRETARY POLICIES VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
def run_secretary_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #secretary algorithm: observe sample values during observational window (no sampling) and save max value seen
    #   then during sampling window, take first sample with value above observed max
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #observe first part of stream
        n_observe = int(n_valid_samples//(k*args.secretary_window))
        max_var = -1
        for day in substream[0:n_observe]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #update threshold if higher value is found
            if var>max_var:
                max_var = var
        
        #iteratve over sampling window
        sample = -1
        for day in substream[n_observe:]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>max_var and sample == -1:
                sample = day
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances

def run_valuemax_secretary_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #valuemax secretaary algorithm: similar to secretary but with two different sampling windows
    #   Phase I: observe sample values during observational window (no sampling) and save max value seen
    #   Phase II: use a threshold calculated using historical max and observed max, sample if above
    #   Phase III: traditional secretary, take first sample with value above observed max
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #get predicted best historically for whole window
        n_sea_hist, _, _ = pred_history.shape
        day_vars = np.zeros((n_sea_hist, len(substream)))
        for sea in range(n_sea_hist):
            for d, day in enumerate(substream):
                day_vars[sea, d] = get_metric(pred_history[sea,day,:], weights, args.policy_metric)
        sea_maxes = np.max(day_vars, axis=1)
        max_pred = np.average(sea_maxes)
        
        #calculate window boundaries
        max_var = -1
        z = -1 / (args.valmax_secretary_c*args.secretary_window)
        lambert_1 = int(math.exp(lambertw(z,-1))*(last_idx-start_idx))
        lambert_0 = int(math.exp(lambertw(z))*(last_idx-start_idx))
        
        #PHASE I: observation window
        for day in substream[0:lambert_1]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #update threshold if higher value is found
            if var>max_var:
                max_var = var
        
        #calculate threshold based on observation and historical max
        sample = -1
        if args.valuemax_confidence == 1:
            conf_param = 0
        else:
            interval = st.t.interval(alpha=args.valuemax_confidence, df=len(sea_maxes)-1, loc=np.mean(sea_maxes), scale=st.sem(sea_maxes))
            conf_param = (interval[1]-interval[0])/2
        max_var_pred = max(max_var, max_pred - conf_param)
        
        #PHASE II: use threshold caluclated from both observed and predicted max
        for day in substream[lambert_1:lambert_0]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>max_var_pred and sample == -1:
                sample = day
            #update threshold if higher value is found
            if var>max_var:
                max_var = var
        
        #PHASE III: revert to secretary threshold still haven't sampled
        for day in substream[lambert_0:]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take first sample which is over the threshold
            if var>max_var and sample == -1:
                sample = day
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances

def run_maxoracle_secretary_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #always pick the highest value sample during the scretary algorithm sampling window
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #determine how often to sample
    k=args.n_samples
    substream_len = n_valid_samples/k
    #create structures to save data
    samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(samples, expert_losses, args.a_weight)
    
    #iterate over all substreams, get one sample per substream
    for i in range(k):
        #get substream
        start_idx = int(i*substream_len)
        last_idx = int((i+1)*substream_len)
        substream = valid_samples[start_idx:last_idx]
        
        #observe first part of stream
        n_observe = int(n_valid_samples//(k*args.secretary_window))
        for day in substream[0:n_observe]:
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
        
        #iterate over all days in substream sampling window
        sample = -1
        max_var = 0
        for day in substream[n_observe:]: 
            #save variance on each day for later plotting
            var = get_metric(pred_labels[day,:], weights, args.policy_metric)
            variances[day] = var
            #take sample with greatest value
            if var>max_var:
                sample = day
                max_var = var
        #sample on last day if no sample has been collected (none were above threshold)
        if sample == -1:
            sample = substream[-1]
        #save samples and update weights accordingly
        samples.append(sample.item())
        weights = get_weights(samples, expert_losses, args.a_weight)
    
    #return data
    return samples, variances


#BASIC POLICIES VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
def run_uniform_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #collect evenly spaced out samples, maximizing time between samples over a single season
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #create structures to save data
    collected_samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(collected_samples, expert_losses, args.a_weight)
    
    #collect evenly spaced samples
    if args.n_samples > 1:
        gap = (n_valid_samples-1)/(args.n_samples - 1)
        collected_samples = np.array([valid_samples[int(i*gap)].item() for i in range(args.n_samples)])
    #if only collecting a single sample, take sample in middle
    else:
        collected_samples = np.array([valid_samples[((n_valid_samples-1)//2)]])

    seen_days = []
    for day in range(n_valid_samples):
        #save variance on each day for later plotting
        var = get_metric(pred_labels[day,:], weights, args.policy_metric)
        variances[day] = var
        #update weights after sample collection
        if day in collected_samples:
            seen_days.append(day)
            weights = get_weights(seen_days, expert_losses, args.a_weight)
    
    #return data
    return collected_samples, variances
    
def run_random_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #randomly select samples
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #create structures to save data
    collected_samples = np.array(random.choices(valid_samples, k=args.n_samples))
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights([], expert_losses, args.a_weight)
    
    seen_days = []
    for day in range(n_valid_samples):
        #save variance on each day for later plotting
        var = get_metric(pred_labels[day,:], weights, args.policy_metric)
        variances[day] = var
        #update weights after sample collection
        if day in collected_samples:
            seen_days.append(day)
            weights = get_weights(seen_days, expert_losses, args.a_weight)
    
    #return data
    return collected_samples, variances
    
def run_baseline_policy(args, actual_labels, pred_labels, expert_losses, pred_history):
    #collect no samples and weight evenly
    
    #get all possible days to collect sample
    valid_samples = np.squeeze(np.argwhere(np.isnan(actual_labels)==0))
    n_valid_samples = len(valid_samples)
    if n_valid_samples == 0:
        print("error: no samples")
        return []
    #create structures to save data
    collected_samples = []
    variances = np.zeros((actual_labels.shape[0]))
    weights = get_weights(collected_samples, expert_losses, args.a_weight)
    
    for day in range(n_valid_samples):
        #save variance on each day for later plotting
        var = get_metric(pred_labels[day,:], weights, args.policy_metric)
        variances[day] = var
    
    #return data
    return np.zeros((args.n_samples)), variances


#HELPER FXNS VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV    
def get_metric(values, weights, version):
    if version in ['weighted_variance_SQ']:
        w_average = np.average(values, weights=weights)
        w_var = 0
        for i in range(len(values)):
            w_var += ((values[i] - w_average)**2) * (weights[i]**2)
        return w_var
    elif version in ['weighted_variance_nonSQ']:
        w_average = np.average(values, weights=weights)
        w_var = 0
        for i in range(len(values)):
            w_var += ((values[i] - w_average)**2) * weights[i]
        return w_var
    
def get_weights(samples,season_losses, a_weight):
    _, n_experts = season_losses.shape
    
    if len(samples) > 0: #normalized exponential weighting based on losses on sample days
        losses = np.sqrt(np.mean(season_losses[samples,:], axis=0))
        weights = [math.exp(-1*a_weight*w) for w in losses]
        total = sum(weights)
        weights = [w/total for w in weights]
    else: #equal weighting if no samples
        weights = [1/n_experts for n in range(n_experts)]
    
    return np.array(weights)
    

#HUMAN EVAL WORK VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#used for Real Time CH sampling (in progress, as mentioned in conclusion of the paper)
def run_human_eval(args):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    
    #model set up and loading
    model = args.nn_model(feature_len, args.n_experts)
    model.to(args.device)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    check_path = os.path.join('.','models', "ColdHardiness", "RealTimeModel", "concat_embedding_setting_leaveoneout_Concord_trial_0.pt")
    checkpoint = torch.load(check_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(args.device)
    
    #set up datasets
    train_dataset = MyDataset(dataset['train'])
    x_train, _, _ = train_dataset[:]
    total_seasons = x_train.shape[0]
    train_batch_size = int(total_seasons/args.n_tasks)
    trainLoader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False) 
    
    loss_dict = dict()
    loss_dict['train'] = dict()
    loss_dict['test'] = dict()

    with torch.no_grad():
        model.eval()
        loss_dict['variances'] = dict()
        loss_dict['samples'] = dict()
        loss_dict['sample_vals'] = dict()
        loss_dict['weights'] = dict()
        loss_dict['preds'] = dict()
        for i, ((x, y, task_id), task) in enumerate(zip(trainLoader,args.data_tasks)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            
            #get model predictions
            for t_id in range(args.n_experts):
                loss_dict['train'][t_id] = dict()
                task_id_torch = (torch.ones((x.shape[0], x.shape[1]))*t_id).type(torch.LongTensor).to(args.device)
                out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
                
                #loop over all seasons
                for p_label in range(len(args.labels)):
                    loss_dict['train'][t_id][p_label] = dict()
                    loss_dict['train'][t_id][p_label]["pred"] = ((out_primary[:,:,p_label])).detach().cpu().numpy()
                    loss_dict['train'][t_id][p_label]["all_losses_MSE"] = (criterion(out_primary[:,:,p_label], y_torch[:, :, p_label])).detach().cpu().numpy()
            
            _,n_days = loss_dict['train'][0][0]["pred"].shape
            preds = np.empty((0,n_days))
            season_losses = np.empty((0,n_days))
            for t_id in range(args.n_experts):
                preds = np.vstack((preds, loss_dict['train'][t_id][0]["pred"]))
                season_losses = np.vstack((season_losses, loss_dict['train'][t_id][0]["all_losses_MSE"]))
            
            variances, samples, weights = run_record_var_policy(args, preds, season_losses)
            loss_dict['variances'][task] = variances
            loss_dict['samples'][task] = samples
            loss_dict['sample_vals'][task] = y
            loss_dict['weights'][task] = weights
            loss_dict['preds'][task] = preds
    
    if args.run_realtime_eval:
        with torch.no_grad():
            model.eval()
            loss_dict['preds_weighted'] = dict()
            loss_dict['preds_unweighted'] = dict()
            #for i, (x, task) in enumerate(zip(dataset['test']['x'],args.data_tasks)):
            for task in args.data_tasks:
                x = dataset['test']['x'][task]
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                
                #get model predictions
                for t_id in range(args.n_experts):
                    loss_dict['test'][t_id] = dict()
                    task_id_torch = (torch.ones((x.shape[0], x.shape[1]))*t_id).type(torch.LongTensor).to(args.device)
                    out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
                    #loop over all seasons
                    for p_label in range(len(args.labels)):
                        loss_dict['test'][t_id][p_label] = dict()
                        loss_dict['test'][t_id][p_label]["pred"] = ((out_primary[:,:,p_label])).detach().cpu().numpy()
                
                n_seasons,n_days = loss_dict['test'][0][0]["pred"].shape
                preds = np.empty((0,n_seasons,n_days))
                for t_id in range(args.n_experts):
                    preds = np.vstack((preds, np.expand_dims(loss_dict['test'][t_id][0]["pred"], axis=0)))
                
                loss_dict['preds_unweighted'][task] = np.average(preds, axis=0)
                weighted_preds = np.empty((n_seasons, n_days))
                for sea in range(n_seasons):    
                    for day in range(n_days):
                        weighted_preds[sea,day] = np.average(preds[:,sea,day], axis=0, weights=loss_dict['weights'][task])
                loss_dict['preds_weighted'][task] = weighted_preds
                    
       
    return loss_dict 


def run_record_var_policy(args, preds, season_losses):
    n_models, n_days = preds.shape
    weights = [1/n_models for n in range(n_models)]
    
    variances = []
    samples = []
    for day in range(n_days):
        val = get_metric(preds[:,day], weights, args.policy_metric)
        variances.append(val)
        if not math.isnan(season_losses[0,day]): #we have a sample collected on this day
            samples.append(day)
            weights = get_weights(samples, np.transpose(season_losses), args.a_weight)
            print(weights)
    
    return variances, samples, weights


#BASE MODEL TRAINING VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV    
def run_experiment(args):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    
    model = args.nn_model(feature_len, args.n_tasks, len(args.labels))

    model.to(args.device)
    
    trainable_params = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])
    print("Trainable Parameters:", trainable_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    criterion2 = nn.BCELoss(reduction='none')
    criterion2.to(args.device)
    
    dir_path = os.path.join('./models', args.dataset_name, args.name, args.current_task_name, ("trial_" + str(args.trial)))
    if (not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    
    if args.use_trained:
        check_path = os.path.join('./models', args.dataset_name, args.name, args.current_task_name, ("trial_" + str(args.trial)), args.experiment+'_setting_'+args.setting+".pt")
        checkpoint = torch.load(check_path)
        print("training",("trial_" + str(args.trial)),checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
    else:
        start_epoch = -1
        best_loss = 99999999999
        loss = 99999999998
        epoch = -1
    
    
    log_dir = os.path.join('./tensorboard/', args.dataset_name, args.name, args.experiment+'_setting_'+args.setting, ("trial_" + str(args.trial)), args.current_task_name)
    writer = SummaryWriter(log_dir)
    
    train_dataset = MyDataset(dataset['train'])
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = MyDataset(dataset['test'])
    testLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
        # Training Loop
        model.train()
        total_losses = 0
        count = 0
        
        print("train w train")
        for i, (x, y, task_id) in enumerate(trainLoader):
            print(task_id)
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            task_id_torch = task_id.to(args.device)
            count += 1
            out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
            
            optimizer.zero_grad()       # zero the parameter gradients
            nan_locs_primary = y_torch.isnan()
            out_primary[nan_locs_primary] = 0
            y_torch = torch.nan_to_num(y_torch)

            loss = 0
            for p_label in range(len(args.labels)):
                losses = criterion(out_primary[:,:,p_label], y_torch[:, :, p_label])[~nan_locs_primary[:,:,p_label]]
                loss += losses.mean()    
                total_losses += losses.mean().item()
            
            loss.backward()             # backward +
            optimizer.step()            # optimize
            
        writer.add_scalar('Train_Loss', total_losses / count, epoch)
        
        # Validation Loop
        with torch.no_grad():
            model.eval()
            total_losses = 0
            count = 0
            for i, (x, y, task_id) in enumerate(testLoader):
                x_torch = x.to(args.device)
                y_torch = y.to(args.device)
                task_id_torch = task_id.to(args.device)
                count += 1
                out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
                
                nan_locs_primary = y_torch.isnan()
                out_primary[nan_locs_primary] = 0
                y_torch = torch.nan_to_num(y_torch)
                
                for p_label in range(len(args.labels)):
                    losses = criterion(out_primary[:,:,p_label], y_torch[:, :, p_label])[~nan_locs_primary[:,:,p_label]]
                    total_losses += losses.mean().item()
                

            loss = total_losses / count
            writer.add_scalar('Val_Loss', loss, epoch)
            
            if epoch % 10 == 0 and loss < best_loss: #save checkpoint
                save_path = os.path.join('./models', args.dataset_name, args.name, args.current_task_name, ("trial_" + str(args.trial)), args.experiment+'_setting_'+args.setting+".pt")

                check_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
                print("mod save",check_dict.keys())
                torch.save(check_dict, save_path)
                print(save_path)
                
                best_loss = loss   
    
    loss_dict = dict()

    if loss < best_loss: #save checkpoint
        save_path = os.path.join('./models', args.dataset_name, args.name, args.current_task_name, ("trial_" + str(args.trial)), args.experiment+'_setting_'+args.setting+".pt")
        check_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }
        print("end save",check_dict.keys())
        print(save_path)
        torch.save(check_dict, save_path)
        best_loss = loss
    
    # Validation Loop
    
    with torch.no_grad():
        model.eval()
        total_losses = [0] * (len(args.labels))
        for i, ((x, y, task_id), task) in enumerate(zip(testLoader,args.data_tasks)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            task_id_torch = task_id.to(args.device)
            out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
            
            nan_locs_primary = y_torch.isnan()
            out_primary[nan_locs_primary] = 0
            y_torch = torch.nan_to_num(y_torch)
            
            loss = []
            for p_label in range(len(args.labels)):
                losses = criterion(out_primary[:,:,p_label], y_torch[:, :, p_label])[~nan_locs_primary[:,:,p_label]]
                loss.append(np.sqrt(losses.mean().item()))
                total_losses[p_label] += losses.mean().item()
            
            loss_dict[task] = loss
    loss_dict['overall'] = np.sqrt(total_losses)

    return loss_dict


def run_eval(args):
    dataset = args.dataset
    feature_len = dataset['train']['x'].shape[-1]
    
    model = args.nn_model(feature_len, args.n_tasks, len(args.labels))

    model.to(args.device)
    
    criterion = nn.MSELoss(reduction='none')
    criterion.to(args.device)
    criterion2 = nn.BCELoss(reduction='none')
    criterion2.to(args.device)
    
    dir_path = os.path.join('./models', args.dataset_name, args.name, args.current_task_name, ("trial_" + str(args.trial)))
    if (not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    
    check_path = os.path.join('./models', args.dataset_name, args.name, args.current_task_name, ("trial_" + str(args.trial)), args.experiment+'_setting_'+args.setting+".pt")
    checkpoint = torch.load(check_path)
    print("eval",str(args.trial),checkpoint.keys())
    print(check_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    train_dataset = MyDataset(dataset['train'])
    x_train, _, _ = train_dataset[:]
    total_seasons = x_train.shape[0]
    train_batch_size = int(total_seasons/args.n_tasks)
    trainLoader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    test_dataset = MyDataset(dataset['test'])
    x_test, _, _ = test_dataset[:]
    total_seasons = x_test.shape[0]
    test_batch_size = int(total_seasons/args.n_tasks)
    testLoader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    loss_dict = dict()
    loss_dict['train'] = dict()
    loss_dict['test'] = dict()

    with torch.no_grad():
        model.eval()
        
        for i, ((x, y, task_id), task) in enumerate(zip(trainLoader,args.data_tasks)):
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            task_id_torch = task_id.to(args.device)
            out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
            
            nan_locs_primary = y_torch.isnan()
            out_primary[nan_locs_primary] = 0
            y_torch = torch.nan_to_num(y_torch)
            
            n_seasons, _, _ = x_torch.shape
            
            loss_dict['train'][task] = dict()
            for p_label in range(len(args.labels)):
                loss_dict['train'][task][p_label] = dict()
                for sea in range(n_seasons):
                    loss_dict['train'][task][p_label][sea] = dict()
                    loss_dict['train'][task][p_label][sea]["pred"] = ((out_primary[sea,:,p_label])[~nan_locs_primary[sea,:,p_label]]).detach().cpu().numpy()
                    loss_dict['train'][task][p_label][sea]["act"] = ((y_torch[sea, :, p_label])[[~nan_locs_primary[sea,:,p_label]]]).detach().cpu().numpy()
                    loss_dict['train'][task][p_label][sea]["loss"] = np.sqrt((criterion(out_primary[sea,:,p_label], y_torch[sea,:,p_label])[~nan_locs_primary[sea,:,p_label]]).detach().cpu().numpy())
                    loss_dict['train'][task][p_label][sea]["measure_dates"] = (~nan_locs_primary[sea,:,p_label]).detach().cpu().numpy()

        for i, ((x, y, task_id), task) in enumerate(zip(testLoader,args.data_tasks)):
            
            x_torch = x.to(args.device)
            y_torch = y.to(args.device)
            task_id_torch = task_id.to(args.device)
            
            out_primary, out_aux, _ = model(x_torch, task_label=task_id_torch)
            
            nan_locs_primary = y_torch.isnan()
            out_primary[nan_locs_primary] = 0
            y_torch = torch.nan_to_num(y_torch)
            
            n_seasons, _, _ = x_torch.shape
            loss_dict['test'][task] = dict()
            for p_label in range(len(args.labels)):
                loss_dict['test'][task][p_label] = dict()
                for sea in range(n_seasons):
                    loss_dict['test'][task][p_label][i*args.batch_size + sea] = dict()
                    loss_dict['test'][task][p_label][i*args.batch_size + sea]["pred"] = ((out_primary[sea,:,p_label])[~nan_locs_primary[sea,:,p_label]]).detach().cpu().numpy()
                    loss_dict['test'][task][p_label][i*args.batch_size + sea]["act"] = ((y_torch[sea, :, p_label])[[~nan_locs_primary[sea,:,p_label]]]).detach().cpu().numpy()
                    loss_dict['test'][task][p_label][i*args.batch_size + sea]["loss"] = np.sqrt((criterion(out_primary[sea,:,p_label], y_torch[sea,:,p_label])[~nan_locs_primary[sea,:,p_label]]).detach().cpu().numpy())
                    loss_dict['test'][task][p_label][i*args.batch_size + sea]["measure_dates"] = (~nan_locs_primary[sea,:,p_label]).detach().cpu().numpy()
    
    return loss_dict

