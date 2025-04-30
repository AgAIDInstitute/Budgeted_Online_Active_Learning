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


#Performs linear interpolation on a pandas Series.
def linear_interp(series: pd.Series, method: str = 'linear') -> pd.Series:
    return series.interpolate(method=method, limit_direction='both')

def remove_na(column_name: str, df: pd.DataFrame) -> None:
    if df[column_name].isna().sum() == 0:
        return
    
    df[column_name] = linear_interp(df[column_name])
    if df[column_name].isna().sum() != 0:
        raise ValueError(f"Interpolation failed: {column_name} still has NaNs.")

#Pads, normalizes, and optionally fills missing labels for given seasons.
def split_and_normalize(args, df: pd.DataFrame, season_max_length: int, seasons: list,
                        fill_train_labels=False, x_mean=None, x_std=None, y_mean=None, y_std=None):
    x_list, y_list = [], []

    for season in seasons:
        # Input features: pad to season_max_length
        _x = df.loc[season, args.features].to_numpy()
        pad_x = np.zeros((season_max_length - len(season), len(args.features)))
        _x = np.vstack((_x, pad_x))

        # Labels: pad with NaNs to match input shape
        _y = df.loc[season, args.labels].to_numpy()
        pad_y = np.full((season_max_length - len(season), len(args.labels)), np.NaN)
        _y = np.vstack((_y, pad_y))

        x_list.append(_x)
        y_list.append(_y)

    x = np.stack(x_list)  # shape: (num_seasons, season_max_length, num_features)
    y = np.stack(y_list)

    # Normalize features
    if x_mean is not None and x_std is not None:
        x = (x - x_mean) / x_std

    # Optionally fill missing labels using a pretrained model
    if fill_train_labels or args.fill_all_traintest_labels:
        from nn.models import single_net as nn_model
        model = nn_model(x.shape[-1]).to(args.device)
        checkpoint = os.path.join(
            './models', args.dataset_name, "train_single", args.current_task_name,
            f"trial_{args.trial}", "single_setting_all.pt"
        )
        model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=False)
        model.eval()

        x_tensor = torch.tensor(x, dtype=torch.float32).to(args.device)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        with torch.no_grad():
            out_primary, out_aux, _ = model(x_tensor)
            out_np = out_primary.detach().cpu().numpy()

        if fill_train_labels:
            nan_mask = np.isnan(y)
            y[nan_mask] = out_np[nan_mask]

        if args.fill_all_traintest_labels:
            y = out_np

    #set y values at time index 252+ to NaN for ColdHardiness, sometimes get early samples for next season that shouldn't be used
    if args.dataset_name == 'ColdHardiness':    
        y[:, 252:, :] = np.NaN 

    # Normalize labels if stats are provided
    if y_mean is not None and y_std is not None:
        y = (y - y_mean) / y_std

    return x, y
    

def data_processing_ColdHardiness(task_file, task_idx, args):
    df = args.task_file_dict[task_file]
    #replace -100 in temp to get accurate mean/std
    for column_name in ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']:
        df[column_name] = df[column_name].replace(-100, np.nan)

    #remove extra data before first dormant season
    first_dormant = df['DORMANT_SEASON'].eq(1).idxmax() #gets first time we start dormancy (1 is dormant day)
    relevant_df = df.loc[first_dormant:].reset_index(drop=False)
    #find transitions from DORMANT_SEASON=0 -> 1
    dormant_shifted = relevant_df['DORMANT_SEASON'].shift(1, fill_value=0)
    season_transition = (relevant_df['DORMANT_SEASON'] == 1) & (dormant_shifted == 0)
    season_labels = season_transition.cumsum()

    #group by transition ID to get contiguous season blocks
    grouped_seasons = (
        relevant_df.groupby(season_labels)
        .apply(lambda g: g['index'].tolist())  # Use original indices
        .tolist()
    )

    #remove invalid season and summer readings
    seasons = [group for group in grouped_seasons if df.loc[group[0], 'SEASON'] != '2006-2007']
    
    #filter valid and extra seasons
    valid_seasons = []
    extra_seasons = []
    for season in seasons:
        season_df = df.loc[season]
        missing_temp = season_df['MIN_AT'].isna().sum()
        valid_lte_count = season_df['LTE50'].notna().sum()
        
        if missing_temp <= int(0.1 * len(season)):
            if valid_lte_count > 0:
                valid_seasons.append(season)
            else:
                extra_seasons.append(season)

    #determine max season length and number
    season_lens = list(map(len, valid_seasons))
    season_max_length = max(season_lens, default=0)
    no_of_seasons = len(valid_seasons)

    #trial-based split into train/test
    trial = args.trial
    if trial == 0:
        test_idx = [0, 1]
    elif trial == 1:
        mid = no_of_seasons // 2
        test_idx = [mid - 1, mid]
    elif trial == 2:
        test_idx = [no_of_seasons - 2, no_of_seasons - 1]
    else:
        test_idx = [trial]

    train_seasons = [season for i, season in enumerate(valid_seasons) if i not in test_idx]
    test_seasons = [season for i, season in enumerate(valid_seasons) if i in test_idx]

    #calc stats for normalization
    valid_idx_train = np.concatenate(train_seasons)
    x_mean = df.loc[valid_idx_train, args.features].mean().to_numpy()
    x_std = df.loc[valid_idx_train, args.features].std().to_numpy()

    #expand training set
    if args.use_all_seasons:
        train_seasons.extend(extra_seasons)
    
    #interpolate missing values after normalization stats are computed
    for feature_col in args.features:
        remove_na(feature_col, df)

    #prepare normalized and padded train/test sets
    x_train, y_train = split_and_normalize(args, df, season_max_length, train_seasons, fill_train_labels=args.fill_train_labels, x_mean=x_mean, x_std=x_std) 
    x_test, y_test = split_and_normalize(args, df, season_max_length, test_seasons, x_mean=x_mean, x_std=x_std)
    
    #add task labels
    task_label_train = torch.full((x_train.shape[0], x_train.shape[1], 1), task_idx, dtype=torch.float)
    task_label_test = torch.full((x_test.shape[0], x_test.shape[1], 1), task_idx, dtype=torch.float)
    return x_train, y_train, x_test, y_test, task_label_train, task_label_test


def data_processing_CropSim(task_file, task_idx, args):
    df = args.task_file_dict[task_file]
    season_transition = (df['DAYS ELAPSED'] == 1)
    season_labels = season_transition.cumsum()

    #group by transition ID to get contiguous season blocks
    grouped_seasons = (
        relevant_df.groupby(season_labels)
        .apply(lambda g: g['index'].tolist())  # Use original indices
        .tolist()
    )
    
    #adjust seasons lengths
    if args.CS_label in ['NAMOUNTRT','NDEMANDST','LAI']:
        seasons = [group[40:] for group in grouped_seasons]
    elif args.CS_label in ['KDEMANDLV','WRT','GRLV']:
        seasons = [group[50:] for group in grouped_seasons]
    elif args.CS_label in ['PTRANSLOCATABLE']:
        seasons = [group[75:] for group in grouped_seasons]
    elif args.CS_label in ['RNUPTAKE','RKUPTAKERT','RPUPTAKELV']:
        if args.dataset_name in ['CropSim_Sorghum']:
            seasons = [group[30:130] for group in grouped_seasons]
        else:
            seasons = [group[30:170] for group in grouped_seasons]
    else:
        seasons = grouped_seasons

    #determine max season length and number
    season_lens = list(map(len, seasons))
    season_max_length = max(season_lens, default=0)

    #trial-based split into train/test        
    first_idx = args.trial
    test_idx = list([first_idx * 6 + i for i in range(6)])
    train_seasons = [season for i, season in enumerate(seasons) if i not in test_idx]
    test_seasons = [season for i, season in enumerate(seasons) if i in test_idx]
    
    #calc stats for normalization
    valid_idx_train = np.concatenate(train_seasons)
    x_mean = df.loc[valid_idx_train, args.features].mean().to_numpy()
    x_std = df.loc[valid_idx_train, args.features].std().to_numpy()

    # do interpolation AFTER you calculate the mean/std
    for feature_col in args.features:  # remove nan and do linear interp.
        remove_na(feature_col, df)

    #prepare normalized and padded train/test sets
    x_train, y_train = split_and_normalize(args, df, season_max_length, train_seasons, x_mean=x_mean, x_std=x_std) 
    x_test, y_test = split_and_normalize(args, df, season_max_length, test_seasons, x_mean=x_mean, x_std=x_std)

    #add task labels
    task_label_train = torch.full((x_train.shape[0], x_train.shape[1], 1), task_idx, dtype=torch.float)
    task_label_test = torch.full((x_test.shape[0], x_test.shape[1], 1), task_idx, dtype=torch.float)
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
    
    x_eval_list = []
    if args.run_realtime_eval: #get last five years of data
        current_year = args.start_date.year
        for i in range(5):
            eval_start_date = pd.to_datetime("07/09/"+str(current_year - i - 1), dayfirst=True)
            eval_end_date = pd.to_datetime("15/05/"+str(current_year - i), dayfirst=True)
            eval_season =df[(df['JULDATE_PST']>=eval_start_date)]
            eval_season =eval_season[(eval_season['JULDATE_PST']<=eval_end_date)]
            
            for feature_col in args.features:  # remove nan and do linear interp.
                remove_na(feature_col, eval_season)
                
            x_eval = (eval_season[args.features]).to_numpy()
            if x_eval.shape[0] > 250: #remove any seasons that are missing data
            #norm_features_idx = np.arange(0, x_mean.shape[0])
                pad_x = np.zeros((252 - x_eval.shape[0], len(args.features)))
                x_eval = np.vstack((x_eval, pad_x))
                x_eval = (x_eval - x_mean) / x_std  # normalize
                x_eval_list.append(x_eval)
    return x, y, np.stack(x_eval_list)
    