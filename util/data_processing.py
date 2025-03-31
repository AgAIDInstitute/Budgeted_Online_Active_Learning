import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import os
from scipy.constants import convert_temperature


class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return self.data_dict['x'].shape[0]

    def __getitem__(self, idx):
        return self.data_dict['x'][idx], self.data_dict['y'][idx], self.data_dict['task_id'][idx]


def linear_interp(x, y, missing_indicator, show=False):
    y = np.array(y)
    if (not np.isnan(missing_indicator)):
        missing = np.where(y == missing_indicator)[0]
        not_missing = np.where(y != missing_indicator)[0]
    else:
        # special case for nan values
        missing = np.argwhere(np.isnan(y)).flatten()
        all_idx = np.arange(0, y.shape[0])
        not_missing = np.setdiff1d(all_idx, missing)

    return np.interp(x, not_missing, y[not_missing])


def remove_na(column_name, df):
    total_na = df[column_name].isna().sum()

    df[column_name] = df[column_name].replace(np.nan, -100)
    df[column_name] = linear_interp(np.arange(df.shape[0]), df[column_name], -100, False)
    if df[column_name].isna().sum() != 0:
        assert False

    return


#TODO: makes this generic
def split_and_normalize(args, _df, season_max_length, seasons, 
                        fill_train_labels=False, x_mean=None, x_std=None, y_mean=None, y_std = None):
    x = []
    y = []
    
    for i, season in enumerate(seasons):
        #get input
        _x = (_df[args.features].loc[season, :]).to_numpy()
        #make all input same length
        _x = np.concatenate((_x, np.zeros((season_max_length - len(season), len(args.features)))), axis=0)

        #add_array = np.zeros((season_max_length - len(season), len(args.labels))) TODO DELETE AFTER CHECK
        #add_array[:] = np.NaN
        
        #get labels and add nans to make all same size
        add_array = np.full((season_max_length - len(season), len(args.labels)), np.NaN)
        _y = _df.loc[season, :][args.labels].to_numpy()
        _y = np.concatenate((_y, add_array), axis=0)
        
        x.append(_x)
        y.append(_y)
    
    #convert to numpy
    x = np.array(x)
    y = np.array(y)
    
    #normalize
    norm_features_idx = np.arange(0, x_mean.shape[0])
    x[:, :, norm_features_idx] = (x[:, :, norm_features_idx] - x_mean) / x_std
    
    
    #Fill with single trained model
    if fill_train_labels or args.fill_all_traintest_labels:
        #load in cultivar model
        feature_len = x.shape[-1]
        from nn.models import single_net as nn_model
        model = nn_model(feature_len)
        model.to(args.device)
        check_path = os.path.join('./models', args.dataset_name, "train_single", args.current_task_name, ("trial_" + str(args.trial)), "single_setting_all.pt")
        model.load_state_dict(torch.load(check_path, map_location=args.device), strict=False)
        model.eval()
        #send x to device
        x_torch = torch.Tensor(x).to(args.device)
        y_torch = torch.Tensor(y)
        #run model on x and get y pred
        out_primary, out_aux, _ = model(x_torch)
        out_numpy = out_primary.detach().cpu().numpy()
        
        if fill_train_labels:
            #replace nan with pred value
            nan_mask = torch.isnan(y_torch)
            y[nan_mask] = out_numpy[nan_mask]
        if args.fill_all_traintest_labels:
            y = out_numpy
        
    y[:,252:,:] = np.NaN #sometimes have a random early sample, want to ignore TODO FIX THIS
    if y_mean is not None:
        norm_features_idx = np.arange(0, y_mean.shape[0])
        y[:, :, norm_features_idx] = (y[:, :, norm_features_idx] - y_mean) / y_std  # normalize

    return x, y

def data_processing_ColdHardiness(task_file, task_idx, args):
    df = args.task_file_dict[task_file]
    #replace -100 in temp to get accurate mean/std
    for column_name in ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']:
        df[column_name] = df[column_name].replace(-100, np.nan)
    
    #this will break based on every switch, could rework this for that? 
    '''
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()
    df = df.groupby('group')
    dfs = []
    for name, data in df:
        dfs.append(data)
    '''
    
    
    total_idx = np.arange(df.shape[0])
    dormant_label = df['DORMANT_SEASON'].to_numpy()
    first_dormant = np.where(dormant_label==1)[0][0]
    relevant_idx = total_idx[first_dormant:]
    dormant_seasons = [df[(df['SEASON']==season_name) & (df['DORMANT_SEASON']==1)].index.to_list() for season_name in list(df['SEASON'].unique())]
    dormant_seasons = [x for x in dormant_seasons if len(x)>0]
    temp_season=list()
    seasons=list()
    for idx in relevant_idx[:-1]:
        temp_season.append(idx)
        if dormant_label[idx]==0 and dormant_label[idx+1]==1:
            if df['SEASON'][idx] != '2007-2008':#remove 2007-2008 season, too much missing
                seasons.append(temp_season)
            temp_season=list()
    #add the last season
    if seasons[-1][0]!=temp_season[0]:
        seasons.append(temp_season)

    #add last index of last
    if seasons[-1][-1]!=relevant_idx[-1]:
       seasons[-1].append(relevant_idx[-1])
    season_lens = [len(season) for season in seasons]
    valid_seasons = list()
    extra_seasons = list()

    for season in seasons:
        #look at locations where we have valid lte50 values, we will remove those seasons from the data which do not contain any lte values
        if len(season)==0:
            continue
        
        missing_temp = df.MIN_AT.iloc[season].isna().sum()
        valid_lte_readings = list(np.array(season)[~np.isnan(df['LTE50'].iloc[season].to_numpy())])
        
        #missing temps are less than 10% of season length and there is atleast one LTE value
        if (missing_temp <= int(0.1*len(season))) and (len(valid_lte_readings)>0):
            valid_seasons.append(season)
        elif(missing_temp <= int(0.1*len(season))):
            extra_seasons.append(season)

    season_lens = [len(season) for season in valid_seasons]
    season_max_length = max(season_lens)
    #season_max_length = args.season_max_len
    no_of_seasons = len(valid_seasons)

    # Heres the part where we select the seasons
    if args.trial == 0:
        test_idx = list([0,1])
    elif args.trial == 1:
        test_idx = list([(no_of_seasons // 2) - 1,(no_of_seasons // 2)])
    elif args.trial == 2:
        test_idx = list([no_of_seasons - 2, no_of_seasons - 1])
    else:
        test_idx = [args.trial]
    train_seasons = list()
    test_seasons = list()
    for season_idx, season in enumerate(valid_seasons):
        if season_idx in test_idx:
            test_seasons.append(season)
            #print("Test:", df['SEASON'][season[0]])
        else:
            train_seasons.append(season)
            #print("Train:", df['SEASON'][season[0]])
    
    valid_idx_train = [x for season in train_seasons for x in season]
    x_mean = df[args.features].iloc[valid_idx_train].mean().to_numpy()
    x_std = df[args.features].iloc[valid_idx_train].std().to_numpy()
    
    if args.use_all_seasons:    
        for season_idx, season in enumerate(extra_seasons):
            train_seasons.append(season)
    
    # do interpolation AFTER you calculate the mean/std
    for feature_col in args.features:  # remove nan and do linear interp.
        remove_na(feature_col, df)
    
    x_train, y_train = split_and_normalize(args, df, season_max_length, train_seasons, fill_train_labels=args.fill_train_labels, x_mean=x_mean, x_std=x_std) 
    x_test, y_test = split_and_normalize(args, df, season_max_length, test_seasons, x_mean=x_mean, x_std=x_std)

    task_label_train = torch.ones(
        (x_train.shape[0], x_train.shape[1], 1))*task_idx
    task_label_test = torch.ones(
        (x_test.shape[0], x_test.shape[1], 1))*task_idx
    return x_train, y_train, x_test, y_test, task_label_train, task_label_test
    

def data_processing_CropSim(task_file, task_idx, args):
    df = args.task_file_dict[task_file]

    total_idx = np.arange(df.shape[0])
    year_label = df['DAYS ELAPSED'].to_numpy()
    
    season_count = 0
    temp_season=list()
    seasons=list()
    for idx in total_idx[:-1]:
        temp_season.append(idx)
        if (year_label[idx+1] == 1) and (season_count<50):
            if args.CS_label in ['NAMOUNTRT','NDEMANDST','LAI']:
                seasons.append(temp_season[40:])
            elif args.CS_label in ['KDEMANDLV','WRT','GRLV']:
                seasons.append(temp_season[50:])
            elif args.CS_label in ['PTRANSLOCATABLE']:
                seasons.append(temp_season[75:])
            elif args.CS_label in ['RNUPTAKE','RKUPTAKERT','RPUPTAKELV']:
                if args.dataset_name in ['CropSim_Sorghum']:
                    seasons.append(temp_season[30:130])
                else:
                    seasons.append(temp_season[30:170])
            else:
                seasons.append(temp_season)
            season_count += 1
            temp_season=list()
    
    #add the last season if haven't reached limit
    if (seasons[-1][0]!=temp_season[0]) and (season_count<50):
        seasons.append(temp_season)
        #add last index of last
        if seasons[-1][-1]!=total_idx[-1]:
            seasons[-1].append(total_idx[-1])

    season_lens = [len(season) for season in seasons]
    season_max_length = max(season_lens)
    #season_max_length = args.season_max_len

    #Heres the part where we select the seasons        
    first_idx = args.trial
    test_idx = list([first_idx * 6 + i for i in range(6)])
    #print(args.trial, test_idx)
    train_seasons = list()
    test_seasons = list()
    #print(len(seasons))
    for season_idx, season in enumerate(seasons):
        if season_idx in test_idx:
            test_seasons.append(season)
            #print("Test:", df['Date'][season[0]])
        else:
            train_seasons.append(season)
            #print("Train:", df['Date'][season[0]])
    
    valid_idx_train = [x for season in train_seasons for x in season]
    x_mean = df[args.features].iloc[valid_idx_train].mean().to_numpy()
    x_std = df[args.features].iloc[valid_idx_train].std().to_numpy()
    
    #y_mean = df[args.labels].iloc[valid_idx_train].mean().to_numpy()
    #y_std = df[args.labels].iloc[valid_idx_train].std().to_numpy()

    # do interpolation AFTER you calculate the mean/std
    for feature_col in args.features:  # remove nan and do linear interp.
        remove_na(feature_col, df)
    
    x_train, y_train = split_and_normalize(args, df, season_max_length, train_seasons, x_mean=x_mean, x_std=x_std) 
    x_test, y_test = split_and_normalize(args, df, season_max_length, test_seasons, x_mean=x_mean, x_std=x_std)#, y_mean=y_mean, y_std=y_std

    task_label_train = torch.ones(
        (x_train.shape[0], x_train.shape[1], 1))*task_idx
    task_label_test = torch.ones(
        (x_test.shape[0], x_test.shape[1], 1))*task_idx
        
    return x_train, y_train, x_test, y_test, task_label_train, task_label_test
    
    
def data_processing_realtimeCH(cultivar_file, cultivar_idx, args):

    df = args.task_file_dict[cultivar_file]
    # replace -100 in temp to get accurate mean/std
    for column_name in ['AVG_AT_F','MIN_AT_F','MAX_AT_F']:
        df[column_name] = df[column_name].replace(-100, np.nan)
    
    for label in ['AVG_AT_F','MIN_AT_F','MAX_AT_F','MIN_DEWPT_F','AVG_DEWPT_F','MAX_DEWPT_F']:
        df[label] = convert_temperature(df[label],'F','C') #convert labels from F to C
    
    df['JULDATE_PST'] = pd.to_datetime(df['JULDATE_PST'].astype(str), dayfirst=False)
    
    dormant_season = df[(df['JULDATE_PST']>=args.start_date)]
    
    normalize_data = df[(df['JULDATE_PST']<args.start_date)]
    
    x_mean = normalize_data[args.features].mean().to_numpy()
    x_std = normalize_data[args.features].std().to_numpy()

    # do interpolation AFTER you calculate the mean/std
    for feature_col in args.features:  # remove nan and do linear interp.
        remove_na(feature_col, dormant_season)
    
    x = (dormant_season[args.features]).to_numpy()
    x = np.expand_dims(x, axis=0)
    norm_features_idx = np.arange(0, x_mean.shape[0])
    x[:, :, norm_features_idx] = (x[:, :, norm_features_idx] - x_mean) / x_std  # normalize
    
    #make sim dimension y filled in nans
    y = np.full((1,x.shape[1],len(args.labels)), np.nan)

    #get samples, if they exist
    #from args.samples_file, look at any STATION_ID rows that match
    sample_df = args.samples_file[(args.samples_file['STATION_ID']==int(cultivar_file))]
    #for row, add lte10, lte50, and lte90
    for i, label in enumerate(args.labels):
        for row in zip(sample_df['DORMANCY_DAY'], sample_df[label]):
            y[0,int(row[0]-1),i] = row[1]
    return x, y