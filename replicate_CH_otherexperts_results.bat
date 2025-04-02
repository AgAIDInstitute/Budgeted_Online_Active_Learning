python main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --run_sampling --run_weighting_eval --policy baseline

FOR %%P IN (uniform,secretary) DO (
    ECHO "policy: %%P"
    FOR %%N IN (2,3) DO (
        ECHO "n samples: %%N"
        python main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --run_sampling --run_weighting_eval --policy %%P --n_samples %%N
    )
)

python results_main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --save_file replicate_CHotherexperts_results_losses --function create_results_file_losses --comp_policies baseline uniform secretary --n_samples_list 2 3 4 5
python results_main.py --name replicate_CHotherexperts_results --other_experts 17 --dataset_name ColdHardiness --save_file replicate_CHotherexperts_results_vars --results_policies uniform secretary prophet_nthreshold empirical_1threshold maxoracle_secretary --function create_results_file_vars --n_samples_list 2 3 4 5
