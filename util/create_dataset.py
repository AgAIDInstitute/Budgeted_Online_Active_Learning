import numpy as np
import torch
from .data_processing import *


#fxn to set up data sets for all tasks
def create_dataset(args):
    x_train_list, y_train_list, x_test_list, y_test_list, task_label_train_list, task_label_test_list = list(), list(), list(), list(), list(), list()
    #process data for each task
    for task_idx, task in enumerate(args.data_tasks):
        #use different task processing fxn for ColdHardiness and CropSim
        if args.dataset_name == "ColdHardiness":
            x_train, y_train, x_test, y_test, task_label_train, task_label_test = data_processing_ColdHardiness(task, task_idx, args)
        else:
            x_train, y_train, x_test, y_test, task_label_train, task_label_test = data_processing_CropSim(task, task_idx, args)
        #update lists
        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        task_label_train_list.append(task_label_train)
        task_label_test_list.append(task_label_test)
    
    #create and return data dictionary
    train_dataset = {'x':torch.Tensor(np.concatenate(x_train_list)),'y':torch.Tensor(np.concatenate(y_train_list)),'task_id':torch.squeeze(torch.Tensor(np.concatenate(task_label_train_list)).long())}
    test_dataset = {'x':torch.Tensor(np.concatenate(x_test_list)),'y':torch.Tensor(np.concatenate(y_test_list)),'task_id':torch.squeeze(torch.Tensor(np.concatenate(task_label_test_list)).long())}
    return {'train':train_dataset, 'test':test_dataset}


#fxn to set up data sets for all tasks, special for realtimeCH bc no train/test split
def create_dataset_realtimeCH(args):
    x_train_list = list()
    y_train_list = list()
    #process data for each task
    for task_idx, task in enumerate(args.data_tasks):
        #process data for each task
        x, y = data_processing_realtimeCH(task, task_idx, args)
        #update lists
        x_train_list.append(x)
        y_train_list.append(y)
    
    #create and return data dictionary
    x_torch = torch.Tensor(np.concatenate(x_train_list))
    y_torch = torch.Tensor(np.concatenate(y_train_list))
    train_dataset = {'x':x_torch, 'y':y_torch, 'task_id':x_torch}
    return {'train':train_dataset}
    