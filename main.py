import argparse
import datetime
import glob
import os
import pickle
from pathlib import Path
import math
import pandas as pd
import torch
import gc

if __name__ == '__main__':
    #main arguments
    parser = argparse.ArgumentParser(description="Arguments to run an experiment, either training a model or running a BOAL experiment")
    parser.add_argument('--experiment', type=str, default="concat_embedding", choices=['single', 'multihead', 'concat_embedding'], help='model type')
    parser.add_argument('--policy', type=str, default="no_policy", choices=['no_policy','record_var','baseline','random','uniform','secretary','valuemax_secretary','maxoracle_secretary','prophet_median','prophet_emax','prophet_2threshold','prophet_nthreshold','maxoracle_prophet','empirical_1threshold'], help='policy to select samples')
    parser.add_argument('--setting', type=str, default="all", choices=['all','leaveoneout','baseline_all'], help='experiment setting')
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S"), help='name of the experiment')
    parser.add_argument('--data_path', type=str, default='./data/', help="dataset location")
    parser.add_argument('--dataset_name', type=str, default='ColdHardiness', choices=['ColdHardiness','CropSim_Wheat','CropSim_Maize','CropSim_Millet','CropSim_Sorghum','RealTimeCH'], help="csv Path")
    parser.add_argument('--results_path', type=str, default='./raw_results/', help="location to save results")
    
    #flag to choose which parts to run
    parser.add_argument('--train_model', action='store_true', help="train expert models")
    parser.add_argument('--preprocess_data', action='store_true', help="run data preprocessing, do if haven't done before")
    parser.add_argument('--run_sampling', action='store_true', help="run a sampling policy and get model weighting")
    parser.add_argument('--run_weighting_eval', action='store_true', help="evaluate weighted model on test set")
    
    #model training arguments
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to run the model for')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size")
    parser.add_argument('--use_trained', action='store_true', help="Train from an existing model, excluding this flag will start training from scratch")
    
    #Sampling Arguments
    #used by all
    parser.add_argument('--n_samples', type=int, default=3, help='Number of samples for policy to collect')
    parser.add_argument('--a_weight', type=float, default=1, help="Constant used when weighting experts")
    #sampling algorithm paramenters
    parser.add_argument('--policy_metric', type=str, default="weighted_variance_nonSQ", choices=['weighted_variance_SQ', 'weighted_variance_nonSQ'], help='Selection criteria used by selection algs')
    parser.add_argument('--secretary_window', type=float, default=math.e, help="Used to determine length of observation window for secretary alg")
    parser.add_argument('--valmax_secretary_c', type=float, default=2, help="Parameter for valuemax secretary alg")
    parser.add_argument('--empirical_metric', type=str, default="mean", choices=['median','mean'], help='Metric used when determining best historical empirical threshold')
    parser.add_argument('--valuemax_confidence', type=float, default=0.9, help="Confidence parameter used to determine threshold in valuemax secretary algorithm")
    #data processing parameters
    parser.add_argument('--use_all_seasons', action='store_true', help="Add seasons with no LTE data to training set, used with fill_train_labels or fill_all_traintest_labels")
    parser.add_argument('--CS_label', type=str, default="", choices=['LAI','DVS','SM','NAmountRT','NdemandST','NAVAIL','WSO','NAMOUNTRT','NDEMANDST','RNUPTAKE','RKUPTAKERT','RPUPTAKELV','KDEMANDLV','PTRANSLOCATABLE','PAVAIL','PERCT','WRT','GRLV'], help='Label to use for CropSim dataset')
    parser.add_argument('--other_experts', type=str, default=None, help="Use a different set of nn experts") 
    parser.add_argument('--fill_train_labels', action='store_true', help="impute missing training labels using single model trained on data")
    parser.add_argument('--fill_all_traintest_labels', action='store_true', help="replace all training and testing labels using single model trained on data")
    
    args = parser.parse_args()
    
    
    #get tasks/features/labels based on dataset
    args.data_path = args.data_path + args.dataset_name + "/"
    if args.dataset_name in ['ColdHardiness']:
        args.n_trials = 3 
        from util.create_dataset import create_dataset
        args.features = [
            'MEAN_AT',    #mean air temp
            'MIN_AT',     #minimum air temp
            'MAX_AT',     #maximum air temp
            'MIN_RH',     #minimum relative humidity
            'AVG_RH',     #average relative humidity
            'MAX_RH',     #maximum relative humidity
            'MIN_DEWPT',  #minimum dew point
            'AVG_DEWPT',  #average dew point
            'MAX_DEWPT',  #maximum dew point
            'P_INCHES',   #precipitation 
            'WS_MPH',     #wind speed
            'MAX_WS_MPH', #max wind speed
        ]
        args.labels = [
            'LTE50'       #cold hardiness
        ]
        args.expert_tasks = [
            'Zinfandel',
            'Concord',
            'Malbec',
            'Barbera',
            'Semillon',
            'Merlot',
            'Chenin Blanc',
            'Riesling',
            'Nebbiolo',
            'Cabernet Sauvignon',
            'Chardonnay',
            'Viognier',
            'Mourvedre',
            'Pinot Gris',
            'Grenache',
            'Syrah',
            'Sangiovese',
            'Sauvignon Blanc'
        ]
        args.eval_tasks =  [
            'Chardonnay',
            'Grenache',
            'Merlot',
            'Mourvedre',
            'Pinot Gris',
            'Sangiovese',
            'Syrah',
            'Viognier'
        ]
        args.task_file_dict = {task: pd.read_csv(glob.glob(args.data_path+'*'+task+'*')[0],low_memory=False) for task in args.expert_tasks}
    elif args.dataset_name in ['RealTimeCH']:
        from util.create_dataset import create_dataset_realtimeCH as create_dataset
        args.start_date = pd.to_datetime("07/09/2024", dayfirst=True)#first day of current dormant season
        args.policy = 'record_var'
        args.features = [
            'AVG_AT_F',
            'MIN_AT_F',  
            'MAX_AT_F',  
            'MIN_REL_HUMIDITY',  
            'AVG_REL_HUMIDITY',  
            'MAX_REL_HUMIDITY',  
            'MIN_DEWPT_F',  
            'AVG_DEWPT_F',  
            'MAX_DEWPT_F',  
            'P_INCHES',
            'WS_MPH',
            'WS_MAX_MPH',  
        ]
        args.labels = [
            'LTE50'
        ]
        args.expert_tasks = [
            'Zinfandel',
            'Malbec',
            'Barbera',
            'Semillon',
            'Merlot',
            'Chenin Blanc',
            'Riesling',
            'Nebbiolo',
            'Cabernet Sauvignon',
            'Chardonnay',
            'Viognier',
            'Mourvedre',
            'Pinot Gris',
            'Grenache',
            'Syrah',
            'Sangiovese',
            'Sauvignon Blanc'
        ]
        args.eval_tasks = ['300207','300026','300254','300133']
        args.task_file_dict = {task: pd.read_csv(glob.glob(args.data_path+'*'+task+'*')[0],low_memory=False) for task in args.eval_tasks}
    elif args.dataset_name in ['CropSim_Wheat','CropSim_Maize','CropSim_Millet','CropSim_Sorghum']:
        args.n_trials = 6
        from util.create_dataset import create_dataset
        args.features = [
            'IRRAD', #Daily solar irradiation
            'TMIN',  #Min daily temperature
            'TMAX',  #Max daily temperature
            'TEMP',	 #Mean daily temperature
            'VAP',	 #Mean daily vapor pressure
            'RAIN',	 #Daily rainfall
            'WIND'   #Mean daily wind speed
        ]
        args.labels = [args.CS_label]
        args.expert_tasks = [str(x) + "_" for x in range(1,21)]
        args.eval_tasks = [str(x) + "_" for x in range(1,16)]
        args.task_file_dict = {task: pd.read_csv(glob.glob(args.data_path+task+'*')[0],low_memory=False) for task in args.expert_tasks}

    #get device (cuda or cpu)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #preprocess dataset, get different datasets to a common form
    if args.preprocess_data:
        if args.dataset_name in ['CropSim_Wheat','CropSim_Maize','CropSim_Millet','CropSim_Sorghum']:
            from experiments.unified_api import run_preprocess_data_CropSim
            data_dict = dict()
            
            #run for each eval task
            for left_out in args.eval_tasks:
                data_dict[left_out] = dict()
                
                for trial in range(args.n_trials):
                    args.trial = trial
                    #get eval task dataset
                    args.data_tasks = list([left_out])
                    args.current_task_name = left_out
                    args.n_tasks = len(args.data_tasks)
                    args.dataset = create_dataset(args)
                    #get expert prediction dataset, removing the eval task to prevent information leakage
                    experts_copy = args.expert_tasks.copy()
                    experts_copy.remove(left_out)
                    args.data_tasks = experts_copy
                    args.n_experts = len(args.data_tasks)
                    args.dataset_expert = create_dataset(args)
                    #get preprocessed dataset
                    data_dict[left_out][trial] = run_preprocess_data_CropSim(args)
        #coldhardiness
        else:
            from experiments.unified_api import run_preprocess_data_CH
            #import expert model
            exec("from nn.models import "+args.experiment+"_net as nn_model")
            args.nn_model = nn_model
            data_dict = dict()
            
            #run for each eval task
            for left_out in args.eval_tasks:
                gc.collect()
                data_dict[left_out] = dict()
                
                for trial in range(args.n_trials):
                    args.trial = trial
                    #get dataset
                    args.data_tasks = list([left_out])
                    args.current_task_name = left_out
                    args.n_tasks = len(args.data_tasks)
                    args.n_experts = len(args.expert_tasks)-1
                    args.dataset = create_dataset(args)
                    #get preprocessed dataset
                    data_dict[left_out][trial] = run_preprocess_data_CH(args)
                
        #save data
        with open(os.path.join(args.data_path, "preprocessed_data_" + str(args.n_trials) + "_trials_" + args.CS_label + "_" + str(args.other_experts) + ".pkl"), 'wb') as f:
            pickle.dump(data_dict, f)
    
    
    #run sampling using preprocessed dataset
    if args.run_sampling:
        #import fxns
        from experiments.unified_api import run_sampling
        exec("from experiments.unified_api import run_"+args.policy+"_policy as policy_fxn")
        args.policy_fxn = policy_fxn
        
        #read in preprocessed
        with open(os.path.join(args.data_path, "preprocessed_data_" + str(args.n_trials) + "_trials_" + args.CS_label + "_" + str(args.other_experts) + ".pkl"),'rb') as f:
            data_dict = pickle.load(f)
        
        sampling_dict = dict()
        #run for each eval task, get sampling using policy
        for left_out in args.eval_tasks:
            sampling_dict[left_out] = dict()
            for trial in range(args.n_trials):
                sampling_dict[left_out][trial] = run_sampling(args, data_dict[left_out][trial])
        
        #save data
        Path(os.path.join(args.results_path, args.dataset_name, args.name)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples)+"_"+str(args.a_weight)+"_"+args.policy_metric+"_sampling.pkl"), 'wb') as f:
            pickle.dump(sampling_dict, f)
    
    
    #evaluate weighted model learned from sampling
    if args.run_weighting_eval:
        from experiments.unified_api import run_weighting_eval
        
        #read in preprocessed
        with open(os.path.join(args.data_path, "preprocessed_data_" + str(args.n_trials) + "_trials_" + args.CS_label + "_" + str(args.other_experts) + ".pkl"),'rb') as f:
            data_dict = pickle.load(f)
        #read in sampling data
        with open(os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples)+"_"+str(args.a_weight)+"_"+args.policy_metric+"_sampling.pkl"),'rb') as f:
            sampling_dict = pickle.load(f)
            
        eval_dict = dict()
        #run for each eval task, evaluate sample-based weighted model on test set
        for left_out in args.eval_tasks:
            eval_dict[left_out] = dict()
            for trial in range(args.n_trials):
                eval_dict[left_out][trial] = run_weighting_eval(args, data_dict[left_out][trial], sampling_dict[left_out][trial])
    
        #save data
        Path(os.path.join(args.results_path, args.dataset_name, args.name)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.results_path, args.dataset_name, args.name, args.labels[0]+"_"+args.policy+"_"+str(args.n_samples)+"_"+str(args.a_weight)+"_"+args.policy_metric+"_eval.pkl"), 'wb') as f:
            pickle.dump(eval_dict, f)
    
    
    #train expert models
    if args.train_model:
        #import needed fxns
        from experiments.unified_api import run_experiment
        from experiments.unified_api import run_eval
        exec("from nn.models import "+args.experiment+"_net as nn_model")
        args.nn_model = nn_model
        overall_loss = dict()
        
        #train a model for a single task
        if args.experiment in ['single']:
            for left_out in args.eval_tasks:
                gc.collect()
                loss_dicts = dict()
                loss_dicts['train'] = dict()
                loss_dicts['eval'] = dict()
                
                #create dataset
                args.data_tasks = list([left_out])
                args.n_tasks = len(args.data_tasks)
                args.current_task_name = left_out
                for trial in range(args.n_trials):
                    args.trial = trial
                    args.dataset = create_dataset(args)
                    #train and evaluate model
                    loss_dicts['train'][args.trial] = run_experiment(args)
                    loss_dicts['eval'][args.trial] = run_eval(args)
                overall_loss[left_out] = loss_dicts
        #train a model for all tasks
        else:
            #leave out single cultivar, used as expert model
            if args.setting in ['leaveoneout']:
                for left_out in args.eval_tasks:
                    gc.collect()
                    loss_dicts = dict()
                    loss_dicts['train'] = dict()
                    loss_dicts['eval'] = dict()
                    
                    #create dataset
                    args.data_tasks = list(set(args.expert_tasks) - set([left_out]))
                    args.n_tasks = len(args.data_tasks)
                    args.current_task_name = left_out
                    for trial in range(args.n_trials):
                        args.trial = trial
                        args.dataset = create_dataset(args)
                        #train and evaluate model
                        loss_dicts['train'][args.trial] = run_experiment(args)
                        loss_dicts['eval'][args.trial] = run_eval(args)
                    overall_loss[left_out] = loss_dicts
            #use all cultivars
            else:
                loss_dicts = dict()
                loss_dicts['train'] = dict()
                loss_dicts['eval'] = dict()
                
                #create dataset
                args.current_task_name = 'all'
                args.data_tasks = list(args.expert_tasks)
                args.n_tasks = len(args.data_tasks)
                for trial in range(args.n_trials):
                    args.trial = trial
                    args.dataset = create_dataset(args)
                    #train and evaluate model
                    loss_dicts['train'][args.trial] = run_experiment(args)
                    loss_dicts['eval'][args.trial] = run_eval(args)
                overall_loss['all'] = loss_dicts
    
        #save results
        Path(os.path.join('./models', args.dataset_name, args.name)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join('./models', args.dataset_name, args.name, args.experiment+'_policy_'+args.policy+'_setting_'+args.setting+"_losses.pkl"), 'wb') as f:
            pickle.dump(overall_loss, f)

    
    #run special evalutaion for real-time (human guided) sampling, mentioned in conclusion of the paper, currently trialing for first time!
    if(args.dataset_name in ['RealTimeCH']):
        exec("from nn.models import "+args.experiment+"_net as nn_model")
        overall_loss = dict()
        args.nn_model = nn_model
        
        #read in sample file
        args.samples_file = pd.read_csv(glob.glob(args.data_path+'*'+'2025_AllStation_LTE'+'*')[0],low_memory=False)
        
        #import necessary functions
        from experiments.unified_api import run_human_eval
        args.policy = 'record_var'
        exec("from experiments.unified_api import run_"+args.policy+"_policy as policy_fxn")
        args.policy_fxn = policy_fxn
        
        gc.collect()
        #data setup
        args.data_tasks = args.eval_tasks
        args.n_tasks = len(args.data_tasks)
        args.dataset = create_dataset(args)
        args.n_experts = len(args.expert_tasks)
        #get results
        overall_loss['eval'] = run_human_eval(args)
        #save results
        Path(os.path.join('./models', args.dataset_name, args.name)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join('./models', args.dataset_name, args.name, args.experiment+'_policy_'+args.policy+'_setting_'+args.setting+"_losses.pkl"), 'wb') as f:
            pickle.dump(overall_loss, f)
        
