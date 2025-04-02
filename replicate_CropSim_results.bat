FOR %%C IN (Maize,Millet,Sorghum,Wheat) DO (
    ECHO "crop: %%C"
	FOR %%L IN (NAVAIL,GRLV) DO (
		ECHO "label: %%L"
		
		python main.py --name replicate_CropSim_results --dataset_name CropSim_%%C --preprocess_data --CS_label %%L
		
		python main.py --name replicate_CropSim_results --dataset_name CropSim_%%C --run_sampling --run_weighting_eval --policy baseline --CS_label %%L
		
		FOR %%P IN (uniform,secretary,prophet_nthreshold,empirical_1threshold,maxoracle_secretary) DO (
			ECHO "policy: %%P"
			FOR %%N IN (2,3,4,5,10) DO (
				ECHO "n samples: %%N"
				python main.py --name replicate_CropSim_results --dataset_name CropSim_%%C --run_sampling --run_weighting_eval --policy %%P --CS_label %%L --n_samples %%N
			)
		)

		python results_main.py --name replicate_CropSim_results --dataset_name CropSim_%%C --save_file replicate_CropSim_results_losses_%%C%%L --function create_results_file_losses --comp_policies baseline uniform secretary --n_samples_list 2 3 4 5 10 --CS_label %%L
		python results_main.py --name replicate_CropSim_results --dataset_name CropSim_%%C --save_file replicate_CropSim_results_vars_%%C%%L --results_policies uniform secretary prophet_nthreshold empirical_1threshold maxoracle_secretary --function create_results_file_vars --n_samples_list 2 3 4 5 10 --CS_label %%L

	)
)