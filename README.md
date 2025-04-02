<!-- ABOUT THE PROJECT -->
## About The Project

This project evaluates a novel approach to budgeted online active learning from finite-horizon data streams with extremely limited labeling budgets. In agricultural applications, such streams might include daily weather data over a growing season, and labels require costly measurements of weather-dependent plant characteristics. Our method integrates two key sources of prior information: a collection of preexisting expert predictors and episodic behavioral knowledge of the experts based on unlabeled data streams. Unlike previous research on online active learning with experts, our work simultaneously considers query budgets, finite horizons, and episodic knowledge, enabling effective learning in applications with severely limited labeling capacity. We demonstrate the utility of our approach through experiments on various prediction problems derived from both a realistic agricultural crop simulator and real-world data from multiple grape cultivars, referred to in this code as **ColdHardiness** and **CropSim**. We give executable files that replicate the results given in the paper. The results show that our method significantly outperforms baseline expert predictions, uniform query selection, and existing approaches that consider budgets and limited horizons but neglect episodic knowledge, even under highly constrained labeling budgets.

<!-- GETTING STARTED -->
## Getting Started

Due to file size constraints, the ColdHardiness data file is split into several files and must be recombined before running any ColdHardiness experiments. This is done by running the following:
* Regular ColdHardiness experiments:
```sh
  python process_CH_files.py --merge_data 
  ```
* Alternate expert set ColdHardiness experiments:
```sh
  python process_CH_files.py --merge_data --other_experts 17
  ```
Note that this process only needs to be completed once. After this point, no further preprocessing must be done for the ColdHardiness data. The CropSim data is given here in its raw format and must be preprocessed. This step is included in the executable files.

<!-- USAGE EXAMPLES -->
## Usage

We give executables to replecate all of the results in the paper, in both .sh and .bat form. The executables are:
* CropSim:
```sh
  replicate_CropSim_results
  ```
* Regular ColdHardiness:
```sh
  replicate_CH_results
  ```
* Alternate expert ColdHardiness:
```sh
  replicate_CHotherexperts_results
  ```

<!-- BUGS -->
## Known Issues

Please note that due to a file processing issue, which has since been fixed, the numerical results given in the paper for the CropSim problem differ from those obtained when running the executable here. However, the relative comparisons stay the same, so the discussion and conclusion given in the paper still hold after the correction.











