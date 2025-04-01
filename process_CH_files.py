import argparse
import os
import pickle

if __name__ == '__main__':
    #main arguments
    parser = argparse.ArgumentParser(description="Arguments to run an experiment, either training a model or running a BOAL experiment")
    parser.add_argument('--data_path', type=str, default='./data/', help="dataset location")
    parser.add_argument('--other_experts', type=str, default=None, help="Use a different set of nn experts") 
    parser.add_argument('--merge_data', action='store_true', help="combine ColdHardiness data into one large file")
    parser.add_argument('--split_data', action='store_true', help="split ColdHardiness data into smaller files")
    args = parser.parse_args()
    
    
    #dataset parameters
    full_data_path = args.data_path + "ColdHardiness/"
    n_trials = 3 
    eval_tasks =  [
        'Chardonnay',
        'Grenache',
        'Merlot',
        'Mourvedre',
        'Pinot Gris',
        'Sangiovese',
        'Syrah',
        'Viognier'
    ]

    #combine smaller files locally to run program
    if args.merge_data:
        data_dict = dict()
        #read in all smaller files
        for task in eval_tasks:
            with open(os.path.join(full_data_path, "preprocessed_data_" + str(n_trials) + "_trials__" + str(args.other_experts) + "_" + task + ".pkl"), 'rb') as f:
                data_dict[task] = pickle.load(f)
        
        #save merged large file
        with open(os.path.join(full_data_path, "preprocessed_data_" + str(n_trials) + "_trials__" + str(args.other_experts) + ".pkl"), 'wb') as f:
            pickle.dump(data_dict, f)
    
    
    #split large data file to be able to upload to GitHub
    if args.split_data:
        #get large dataset
        with open(os.path.join(full_data_path, "preprocessed_data_" + str(n_trials) + "_trials__" + str(args.other_experts) + ".pkl"), 'rb') as f:
            data_dict = pickle.load(f)
        
        #save each task in a different file
        for task in eval_tasks:
            with open(os.path.join(full_data_path, "preprocessed_data_" + str(n_trials) + "_trials__" + str(args.other_experts) + "_" + task + ".pkl"), 'wb') as f:
                pickle.dump(data_dict[task], f)