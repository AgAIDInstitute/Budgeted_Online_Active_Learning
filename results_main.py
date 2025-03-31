import argparse
import datetime

#set up parser to read command line arguments
parser = argparse.ArgumentParser(description="Command line arguments for results processing")

#choose which results function to run
parser.add_argument('--function', type=str, default=None, choices=['create_results_file_losses', 'create_results_file_vars','plot_paper_modelpred', 'plot_preds', 'plot_variances'], help='results analysis to run')

#path arguments
parser.add_argument('--data_path', type=str, default='./data/', help="dataset location")
parser.add_argument('--results_path', type=str, default='./raw_results/', help="location to save results")
parser.add_argument('--plots_path', type=str, default='./plots/', help="location to save plots")
parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S"), help='name of the experiment')
parser.add_argument('--save_file', type=str, default='out_file', help="data save file name")

#dataset arguments
parser.add_argument('--dataset_name', type=str, default='ColdHardiness', choices=['ColdHardiness','CropSim_Wheat','CropSim_Maize','CropSim_Millet','CropSim_Sorghum','RealTimeCH'], help="csv Path")
parser.add_argument('--CS_label', type=str, default="", choices=['LAI','DVS','SM','NAmountRT','NdemandST','NAVAIL','WSO','NAMOUNTRT','NDEMANDST','RNUPTAKE','RKUPTAKERT','RPUPTAKELV','KDEMANDLV','PTRANSLOCATABLE','PAVAIL','PERCT','WRT','GRLV'], help='Label to use for CropSim dataset')
parser.add_argument('--a_weight', type=float, default=1, help="Constant used when weighting experts")
parser.add_argument('--policy_metric', type=str, default="weighted_variance_nonSQ", choices=['weighted_variance_SQ', 'weighted_variance_nonSQ'], help='Selection criteria used by selection algs')
parser.add_argument('--other_experts', type=str, default=None, help="Use a different set of nn experts") 
parser.add_argument('--policy', type=str, default="no_policy", choices=['no_policy','record_var','baseline','random','uniform','secretary','valuemax_secretary','maxoracle_secretary','prophet_median','prophet_emax','prophet_2threshold','prophet_nthreshold','maxoracle_prophet','empirical_1threshold'], help='policy to select samples')

#all experiments iteration lists
parser.add_argument('--comp_policies', nargs="*", type=str, default=["baseline"], choices=['baseline','random','uniform','secretary','valuemax_secretary','maxoracle_secretary','prophet_median','prophet_emax','prophet_2threshold','prophet_nthreshold','maxoracle_prophet','empirical_1threshold'], help='policy to select samples')
parser.add_argument('--results_policies', nargs="*", type=str, default=["uniform","secretary","prophet_nthreshold","empirical_1threshold"], choices=['baseline','random','uniform','secretary','valuemax_secretary','maxoracle_secretary','prophet_median','prophet_emax','prophet_2threshold','prophet_nthreshold','maxoracle_prophet','empirical_1threshold'], help='policy to select samples')
parser.add_argument('--n_samples_list', nargs="*", type=int, default=[], help='Number of samples for policy to collect')

#plotting flags
parser.add_argument('--plot_act', action='store_true', help="include actual, groundtruth data in the plot")
parser.add_argument('--plot_samples', action='store_true', help="include samples in the plot")


#get args from parser
args = parser.parse_args()

#set dataset specific parameters
if args.dataset_name in ['ColdHardiness']:
        args.n_trials = 3 
        args.labels = [
            'LTE50'
        ]
        args.eval_tasks =  [
            'Chardonnay',
            'Chenin Blanc',
            'Grenache',
            'Malbec',
            'Merlot',
            'Mourvedre',
            'Pinot Gris',
            'Sangiovese',
            'Sauvignon Blanc',
            'Syrah',
            'Viognier'
        ]
elif args.dataset_name in ['CropSim_Wheat','CropSim_Maize','CropSim_Millet','CropSim_Sorghum']:
    args.n_trials = 6
    args.labels = [args.CS_label]
    args.eval_tasks = [str(x) + "_" for x in range(1,16)]

#import and execute desired results function
exec("from results_functions import "+args.function+" as results_function")
results_function(args)