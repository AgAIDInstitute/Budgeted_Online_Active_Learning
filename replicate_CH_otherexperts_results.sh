#!/bin/sh

# ensure processed data files have been merged prior to running

# get baseline results (does not sample)
python main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --run_sampling --run_weighting_eval --policy baseline

# run sampling for all sampling methods and number of samples
for p in {uniform,secretary,prophet_nthreshold,empirical_1threshold,maxoracle_secretary}
do
    echo "policy: $p"
    for n in {2,3,4,5}
    do
        echo "n samples: $n"
        python main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --run_sampling --run_weighting_eval --policy $p --n_samples $n
    done
done

# save all results to csv
python results_main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --save_file replicate_CHotherexperts_results_losses --function create_results_file_losses --comp_policies baseline uniform secretary --n_samples_list 2 3 4 5
python results_main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --save_file replicate_CHotherexperts_results_vars --results_policies uniform secretary prophet_nthreshold empirical_1threshold maxoracle_secretary --function create_results_file_vars --n_samples_list 2 3 4 5
