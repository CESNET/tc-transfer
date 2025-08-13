# Traffic Classification Transfer Learning

This repository contains the code for the transfer learning experiments from the manuscript *Universal Embedding Function for Traffic Classification via QUIC Domain Recognition Pretraining: A Transfer Learning Success*.

To summarize, this project:
 
- Uses a pretrained model `30pktTCNET_256` from [CESNET Models](https://github.com/CESNET/cesnet-models).
- Evaluates how well the model generalizes by transferring it to seven traffic classification (TC) datasets:

  - ISCXVPN2016
  - MIRAGE19
  - MIRAGE22
  - UTMOBILENET21
  - UCDAVIS19
  - CESNET-TLS22
  - AppClassNet

- Implements three transfer learning methods: k-NN, linear probing, and model fine-tuning, along with training from scratch and an input-space baseline for comparison.
- The datasets cover ten downstream tasks in total. **Our fine-tuning approach surpasses SOTA performance on nine of them.** The hyperparameters used to achieve the best results are listed in `conf/best`.

## How to use

1. Prepare an environment using pip or conda, installing the dependencies listed in  `./requirements`.
2. Download all datasets

   - MIRAGE19, MIRAGE22, UTMOBILENET21, and UCDAVIS19 datasets are obtained from the [tcbench framework](https://github.com/tcbenchstack/tcbench). Follow their instructions for preparing the datasets.
   - CESNET-TLS22 is accessed through [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo). It will download automatically on first use.
   - Download AppClassNet from [figshare](https://figshare.com/articles/dataset/AppClassNet_-_A_commercial-grade_dataset_for_application_identification_research/20375580).
   - For ISCXVPN2016, we use a version provided by Alfredo Nascita. You can contact him at alfredo[dot]nascita[at]unina[dot]it. Before using this version, process it with `scripts/preprocess_iscx_dataset.py`.

3. Update `conf/local-vars.yaml`. Specify a local folder containing AppClassNet and ISCXVPN2016 datasets, a temp folder, and a wandb project name (integration with `wandb` is currently mandatory).

4. Experiments are configured with [Hydra](https://hydra.cc/docs/intro/). Multiple configs are prepared in `./conf`. To run an experiment with a given config, use `python -m experiment_wrapper.do_experiment --config-name local-config.yaml`. 


#### Results

The results of experiments are saved in `$temp_dir/results`. The final results presented in the manuscript are available in `scripts/final-results`. See `scripts/explore_results.ipynb` for a summarization of the results in tables. The best found hyperparameteres for each dataset can be found in `conf\best`.
