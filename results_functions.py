import pickle
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import csv
import scipy.stats as st
import math


#helper fxn for create_results_file_losses, returns lists of losses and Wilcoxan test results
def get_comp_losses_Wilcoxan(args, baseline_policy=None):
    #if comparison policy is not set, compare to self
    if baseline_policy is None:
        baseline_policy = args.policy
    
    #set up return lists
    avg_losses = list()
    avg_diff_losses = list()
    t_stats = list()
    p_vals = list()
    all_losses = list()
    all_baseline_losses = list()
    
    #read in evaluation and comparison files
    eval_file = os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples)+"_"+str(args.a_weight)+"_"+args.policy_metric+"_eval.pkl")
    if baseline_policy == "baseline":
        baseline_eval_file = os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+baseline_policy+"_3_"+str(args.a_weight)+"_"+args.policy_metric+"_eval.pkl")
    else:
        baseline_eval_file = os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+baseline_policy+"_"+str(args.n_samples)+"_"+str(args.a_weight)+"_"+args.policy_metric+"_eval.pkl")
    #files may not exist, must handle fnf exception
    try:
        with open(eval_file,'rb') as f:
            eval_dict = pickle.load(f)
            
        with open(baseline_eval_file,'rb') as f:
            baseline_eval_dict = pickle.load(f)
    #return empty results (list of -1) if files cannot be found
    except Exception as e:
        print(e)
        empty_list = [-1] * (len(args.eval_tasks)+1)
        return empty_list, empty_list, empty_list, empty_list
        
    #iterate over all tasks, trials, and seasons, adding results for each
    for task in args.eval_tasks:
        #task loss list used to get task average and calculate Wilcoxan
        baseline_loss_list = list()
        loss_list = list()
        diff_loss_list = list()
        
        for trial in eval_dict[task].keys():
            n_sea, _ = eval_dict[task][trial]["losses"].shape
            for sea in range(n_sea):
                loss = eval_dict[task][trial]["losses"][sea,0]
                baseline_loss = baseline_eval_dict[task][trial]["losses"][sea,0]
                #save losses, if they exist
                if (not math.isnan(loss)) and (not math.isnan(baseline_loss)):
                    diff_loss_list.append(loss - baseline_loss)
                    baseline_loss_list.append(baseline_loss)
                    loss_list.append(loss)
        #save losses for combined Wilcoxan
        all_baseline_losses.extend(baseline_loss_list)
        all_losses.extend(loss_list)
        #save average losses to return
        avg_diff_losses.append(statistics.mean(diff_loss_list))
        avg_losses.append(statistics.mean(loss_list))
        #caluclate Wilcoxan if baseline policy is not the same as policy, otherwise use 1 as placeholder
        if(baseline_policy == args.policy):
            t_stat, p_value = 1, 1
        else:
            t_stat, p_value = st.wilcoxon(baseline_loss_list, loss_list, alternative="greater")
        t_stats.append(t_stat)
        p_vals.append(p_value)
    
    #get final average and Wilcoxan over all tasks
    avg_losses.append(statistics.mean(avg_losses))
    avg_diff_losses.append(statistics.mean(avg_diff_losses))
    
    if(baseline_policy == args.policy):
        t_stat, p_value = 1, 1
    else:
        t_stat, p_value = st.wilcoxon(all_baseline_losses, all_losses, alternative="greater")
    t_stats.append(t_stat)
    p_vals.append(p_value)
    
    #return final results
    return avg_losses, avg_diff_losses, t_stats, p_vals


#helper fxn for create_results_file_vars, returns list of variance results
def get_variances(args):
    #set up return list
    avg_vars = list()
    
    #read in file, file may not exist, must handle fnf exception
    sample_file = os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples)+"_"+str(args.a_weight)+"_"+args.policy_metric+"_sampling.pkl")
    try:
        with open(sample_file,'rb') as f:
            sampling_dict = pickle.load(f)
    #return empty results (list of -1) if files cannot be found
    except Exception as e:
        print(e)
        return [-1] * (len(args.eval_tasks)+1)
    
    #iterate over all tasks, trials, and seasons, adding results for each
    for task in args.eval_tasks:
        #task vars list used to get task average
        task_vars = list()
        
        for trial in sampling_dict[task].keys():
            n_sea, _, _ = sampling_dict[task][trial]["variances"].shape
            variances = sampling_dict[task][trial]["variances"]
            sample_days = sampling_dict[task][trial]["sample_days"]
            
            for sea in range(n_sea):
                #save vars from days samples were collected
                for day in sample_days[sea,:,0]:
                    task_vars.append(variances[sea,int(day),0])
        
        avg_vars.append(statistics.mean(task_vars))
    
    #get final average over all tasks
    avg_vars.append(statistics.mean(avg_vars))
    #return final results
    return avg_vars         


#fxn to generate all experiment losses csv
def create_results_file_losses(args):
    #SET USING ARGS:
    #function = create_results_file_losses
    #comp_policies
    #n_samples_list
    #results_policies
    #dataset_name
    #CS_label (CropSim)
    #name (optional)
    #results_path (optional)

    #open csv and create writer to write results with
    with open(args.save_file+".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        
        #iterate over all policies being compared to
        for comp_policy in args.comp_policies:
            writer.writerow(comp_policy)
            #set up data save lists
            names, all_losses, all_diff_losses, all_tstats, all_pvals = list(), list(), list(), list(), list()
            
            #get results for baseline (sampling does not matter)
            names.append("baseline")
            args.policy = "baseline"
            args.n_samples = 3
            avg_losses, avg_diff_losses, tstats, pvals = get_comp_losses_Wilcoxan(args, comp_policy)
            #save results
            all_losses.append(avg_losses)
            all_diff_losses.append(avg_diff_losses)
            all_tstats.append(tstats)
            all_pvals.append(pvals)
            
            #get results for all number of samples collect and considered policies
            for n_samples in args.n_samples_list:
                for policy in args.results_policies:
                    names.append(str(n_samples)+"_sample_"+policy)
                    args.policy = policy
                    args.n_samples = n_samples
                    avg_losses, avg_diff_losses, tstats, pvals = get_comp_losses_Wilcoxan(args, comp_policy)
            
                    all_losses.append(avg_losses)
                    all_diff_losses.append(avg_diff_losses)
                    all_tstats.append(tstats)
                    all_pvals.append(pvals)
            
            #write transposed rows to csv
            writer.writerow(names)
            writer.writerows(zip(*all_losses))
            writer.writerows(zip(*all_diff_losses))
            writer.writerows(zip(*all_tstats))
            writer.writerows(zip(*all_pvals))


#fxn to generate all experiment variances csv
def create_results_file_vars(args):
    #SET USING ARGS:
    #function = create_results_file_vars
    #n_samples_list
    #results_policies
    #dataset_name
    #CS_label (CropSim)
    #name (optional)
    #results_path (optional)
    
    #open csv and create writer to write results with
    with open(args.save_file+".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        names, all_vars = list(), list()
        
        #get results for all number of samples collect and considered policies
        for n_samples in args.n_samples_list:
            for policy in args.results_policies:
                names.append(str(n_samples)+"_sample_"+policy)
                args.policy = policy
                args.n_samples = n_samples
                avg_vars = get_variances(args)
                all_vars.append(avg_vars)
        
        #write transposed rows to csv
        writer.writerow(names)
        writer.writerows(zip(*all_vars))


#fxn to generate the expert prediction plots given in the paper (Figures 1,2)
def plot_paper_modelpred(args):
    #SET USING ARGS:
    #function = plot_paper_modelpred
    #dataset_name
    #CS_label (CropSim)
    #name (optional)
    #data_path (optional)
    #plot_path (optional)
    #other_experts (ColdHardiness, optional)
    
    #formatting to make text visible in the paper
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    #load in data file with all expert predictions
    with open(os.path.join(args.data_path, args.dataset_name, "preprocessed_data_" + str(args.n_trials) + "_trials_" + args.CS_label + "_" + str(args.other_experts) + ".pkl"), 'rb') as f:
        data_dict = pickle.load(f)
    
    #check if image save directory exists, if not make directory
    Path(os.path.join(args.plots_path, args.dataset_name, args.name,"paper_plots")).mkdir(parents=True, exist_ok=True)
    
    #iterate over all tasks, trials, and seasons, generating one plot per combination
    for task in args.eval_tasks:
        for trial in range(args.n_trials):
            preds = data_dict[task][trial]["test_preds"]
            n_sea, n_days, _, n_experts = preds.shape
            for sea in range(n_sea):
                #create figure and plot expert prections
                fig = plt.figure(figsize =(7, 4))
                x = range(n_days)
                for t_id in range(n_experts):
                    plt.plot(x, preds[sea,:,0,t_id])
                
                #add dataset specific axis labels and limits
                if args.dataset_name == "ColdHardiness":
                    plt.ylabel("Cold Hardiness (LTE)")
                    plt.ylim(-30,-5)
                    plt.xlim(0,252)
                    plt.xticks([24,55,85,116,147,175,206,236], ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
                else:
                    plt.ylabel(args.CS_label)
                    plt.xlabel("Day")
                plt.tight_layout()
                
                #save and close plot
                plt.savefig(os.path.join(args.plots_path, args.dataset_name, args.name, "paper_plots", args.labels[0] + "_" + task + "_trial_" + str(trial) + "_sea_" + str(sea) + ".png"))
                plt.close()


#fxn to generate expert prediction plots, optionally adding the actual value and the days sampled
def plot_preds(args):
    #SET USING ARGS:
    #function = plot_preds
    #dataset_name
    #CS_label (CropSim)
    #name (need if plotting samples)
    #args.policy (need if plotting samples)
    #args.n_samples (need if plotting samples)
    #plot_act (optional)
    #plot_samples (optional)
    #data_path (optional)
    #plot_path (optional)
    #other_experts (ColdHardiness, optional)
    
    
    #read in data dict
    with open(os.path.join(args.data_path, args.dataset_name, "preprocessed_data_" + str(args.n_trials) + "_trials_" + args.CS_label + "_" + str(args.other_experts) + ".pkl"),'rb') as f:
        data_dict = pickle.load(f)
    #read in sampling dict, if needed
    if args.plot_samples:
        with open(os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples_list[0])+"_"+str(args.a_weight)+"_"+args.policy_metric+"_sampling.pkl"),'rb') as f:
            sampling_dict = pickle.load(f)
    
    #check if image save directory exists, if not make directory
    Path(os.path.join(args.plots_path, args.dataset_name, args.name,"pred_plots")).mkdir(parents=True, exist_ok=True)
    
    #iterate over all tasks, trials, and seasons, generating one plot per combination
    for task in args.eval_tasks:
        for trial in range(args.n_trials):
            preds = data_dict[task][trial]["train_preds"]
            actual = data_dict[task][trial]["train_actual"]
            n_sea, n_days, _, n_experts = preds.shape
            for sea in range(n_sea):
                #create figure and plot expert prections
                fig = plt.figure(figsize =(7, 4))
                x = range(n_days)
                for t_id in range(n_experts):
                    plt.plot(x, preds[sea,:,0,t_id])
                
                #plot actual values, if requested
                if args.plot_act:
                    plt.plot(x, actual[sea,:,0], color="k", label="Actual value")
                    plt.legend()
                #plot sample days, if requested
                if args.plot_samples:
                    for sample in sampling_dict[task][trial]["sample_days"][sea]:
                        line = plt.axvline(x = sample, color = 'b')
                    line.set_label('Sample collected')#only include label once
                    plt.legend()
                
                #add dataset specific axis labels and limits, plot formatting
                if args.dataset_name == "ColdHardiness":
                    plt.ylabel("Cold Hardiness (LTE)")
                    plt.ylim(-30,-5)
                    plt.xlim(0,252)
                    plt.xticks([24,55,85,116,147,175,206,236], ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
                else:
                    plt.ylabel(args.CS_label)
                    plt.xlabel("Day")
                title = args.labels[0] + "_" + task + "_trial_" + str(trial) + "_sea_" + str(sea)
                plt.title(title)
                
                #save and close plot
                plt.savefig(os.path.join(args.plots_path, args.dataset_name, args.name,"pred_plots",title + str(args.plot_act) + str(args.plot_samples) + ".png"))
                plt.close()
    
  
#fxn to generate expert variance plots, optionally adding the days sampled
def plot_variances(args):
    #SET USING ARGS:
    #function = plot_variances
    #dataset_name
    #name 
    #args.policy
    #args.n_samples
    #plot_samples (optional)
    #CS_label (CropSim)
    #data_path (optional)
    #plot_path (optional)
    
    
    #read in sampling dict
    with open(os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples_list[0])+"_"+str(args.a_weight)+"_"+args.policy_metric+"_sampling.pkl"),'rb') as f:
        sampling_dict = pickle.load(f)
    
    #check if image save directory exists, if not make directory
    Path(os.path.join(args.plots_path, args.dataset_name, args.name,"variance_plots",args.policy)).mkdir(parents=True, exist_ok=True)
    
    #iterate over all tasks, trials, and seasons, generating one plot per combination
    for task in args.eval_tasks:
        for trial in range(args.n_trials):
            variances = sampling_dict[task][trial]["variances"]
            samples = sampling_dict[task][trial]["sample_days"]
            n_sea, n_days, _ = variances.shape
            for sea in range(n_sea):
                #create figure and plot variance
                fig = plt.figure(figsize =(7, 4))
                x = range(n_days)
                plt.plot(x, variances[sea,:,0], color = "k", label = "Variance")
                
                #plot sample days, if requested
                if args.plot_samples:
                    for sample in samples[sea]:
                        line = plt.axvline(x = sample, color = 'b')
                    line.set_label('Sample collected')#only include label once
                    
                if args.policy in ["secretary", "maxoracle_secretary"]: #secretary, obs zone
                    k=args.n_samples_list[0]
                    substream_len = 252/k if args.dataset_name == "ColdHardiness" else n_days/k
                    first_idxs = [i*substream_len for i in range(k)]
                    n_observe = int(substream_len//(math.e))
                    for idx in first_idxs:
                        obs_window = plt.axvspan(idx, idx+n_observe, color='0.5', alpha=0.5)
                    obs_window.set_label('Observation window')#only include label once
                    
                elif args.policy in ["prophet_nthreshold", "empirical_1threshold", "maxoracle_prophet","baseline"]: #prophet, use whole zone, just do lines
                    k=args.n_samples_list[0]
                    substream_len = 252/k if args.dataset_name == "ColdHardiness" else n_days/k
                    last_idxs = [(i+1)*substream_len - 0.5 for i in range(k)]
                    for idx in last_idxs[:-1]:
                        line = plt.axvline(x = idx, color = '0.6')
                    line.set_label('Sample window boundary')#only include label once
                
                #add dataset specific axis labels and limits, plot formatting
                plt.ylabel("Variance")
                if args.dataset_name == "ColdHardiness":
                    plt.xlim(0,252)
                    plt.xticks([24,55,85,116,147,175,206,236], ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
                else:
                    plt.xlabel("Day")
                title = args.labels[0] +"_"+args.policy+"_"+str(args.n_samples_list[0])+ "_" + task + "_trial_" + str(trial) + "_sea_" + str(sea)
                plt.title(title)
                plt.legend()
                
                #save and close plot
                plt.savefig(os.path.join(args.plots_path, args.dataset_name, args.name,"variance_plots",args.policy,title + str(args.plot_act) + str(args.plot_samples) + ".png"))
                plt.close()