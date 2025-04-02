#!/bin/sh

# do all for each crop and label combination
for crop in {Maize,Millet,Sorghum,Wheat}
do
    echo "crop: $crop"
	for label in {NAVAIL,GRLV}
	do
		echo "label: $label"
		
		# preprocess data
		python main.py --name replicate_CropSim_results --dataset_name CropSim_${crop} --preprocess_data --CS_label $label
		
		# get baseline results (does not sample)
		python main.py --name replicate_CropSim_results --dataset_name CropSim_${crop} --run_sampling --run_weighting_eval --policy baseline --CS_label $label
		
		# run sampling for all sampling methods and number of samples
		for p in {uniform,secretary,prophet_nthreshold,empirical_1threshold,maxoracle_secretary}
		do
			echo "policy: $p"
			for n in {2,3,4,5,10}
			do
				echo "n samples: $n"
				python main.py --name replicate_CropSim_results --dataset_name CropSim_${crop} --run_sampling --run_weighting_eval --policy $p --CS_label $label --n_samples $n
			done
		done

		# save all results to csv
		python results_main.py --name replicate_CropSim_results --dataset_name CropSim_${crop} --save_file replicate_CropSim_results_losses_${crop}${label} --function create_results_file_losses --comp_policies baseline uniform secretary --n_samples_list 2 3 4 5 10 --CS_label $label
		python results_main.py --name replicate_CropSim_results --dataset_name CropSim_${crop} --save_file replicate_CropSim_results_vars_${crop}${label} --results_policies uniform secretary prophet_nthreshold empirical_1threshold maxoracle_secretary --function create_results_file_vars --n_samples_list 2 3 4 5 10 --CS_label $label

	done
done