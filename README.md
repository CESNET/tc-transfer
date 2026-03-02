# Traffic Classification Transfer Learning

This project provides code to reproduce the transfer learning experiments from *Universal Embedding Function for Traffic Classification via QUIC Domain Recognition Pretraining: A Transfer Learning Success*, [https://doi.org/10.48550/arXiv.2502.12930](https://doi.org/10.48550/arXiv.2502.12930).

This project:
 
- Loads the pretrained model `30pktTCNET_256` from [CESNET Models](https://github.com/CESNET/cesnet-models), with the pretraining procedure detailed in the publication.
- Transfers the model to seven well-known traffic classification datasets (ten downstream tasks in total) to evaluate how well it generalizes:

  - ISCXVPN2016
  - MIRAGE19
  - MIRAGE22
  - UTMOBILENET21
  - UCDAVIS19
  - CESNET-TLS22
  - AppClassNet

- Implements three transfer learning methods:

  - k-NN
  - Linear probing
  - Full model fine-tuning

- Includes training from scratch and an input-space baseline for comparison.

## How to Use

1. Install the dependencies listed in `./requirements/pip-requirements.txt`. If you are using Windows, you can create the Conda environment from `./requirements/environment-windows.yml`. Ensure that PyTorch is installed with CUDA support. Install [faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) either as `faiss-cpu` or `faiss-gpu` (pip packages are also available).
2. Download all datasets

    - MIRAGE19, MIRAGE22, UTMOBILENET21, and UCDAVIS19 are obtained from the [tcbench framework](https://github.com/tcbenchstack/tcbench). Follow the tcbench [instructions](https://tcbenchstack.github.io/tcbench/datasets/install/) to install the datasets.
    - CESNET-TLS22 is accessed via [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo). It is downloaded automatically on first use.
    - Download AppClassNet from [figshare](https://figshare.com/articles/dataset/AppClassNet_-_A_commercial-grade_dataset_for_application_identification_research/20375580).
    - For ISCXVPN2016, we used a version provided by Alfredo Nascita. You can contact him at alfredo[dot]nascita[at]unina[dot]it. Before use, process the dataset with `scripts/preprocess_iscx_dataset.py`.

3. Update `conf/local-vars.yaml` and provide:

    - A local folder containing the AppClassNet and ISCXVPN2016 datasets
    - A temporary directory for experiment outputs
    - A `wandb` project name (`wandb` integration for experiment tracking is currently mandatory)

4. Experiments are configured with [Hydra](https://hydra.cc/docs/intro/), with configuration files located in `./conf`. To run an experiment with a specific configuration, use `python -m experiment_wrapper.do_experiment --config-name local-config.yaml`.

## Results

The results of the experiments are saved in `$temp_dir/results`. The final results presented in the publication are available in `scripts/final-results`. For an overview of the results, see [`scripts/explore_results.ipynb`](https://github.com/CESNET/tc-transfer/blob/main/scripts/explore_results.ipynb). The best model fine-tuning hyperparameters for each dataset can be found in `conf/best`.

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{Luxemburk2026Universal,
  author={Luxemburk, Jan and Hynek, Karel and Plný, Richard and Čejka, Tomáš},
  journal={IEEE Transactions on Network and Service Management},
  title={Universal Embedding Function for Traffic Classification via QUIC Domain Recognition Pretraining: A Transfer Learning Success},
  year={2026},
  volume={23},
  pages={1647-1663},
  doi={10.1109/TNSM.2025.3642984}
}
```