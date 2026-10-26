# GTAFL

**GTAFL: Addressing Test-Agnostic Long-Tailed Federated Learning in Medical Image**

> 🎉 Accepted by **IEEE Journal of Biomedical and Health Informatics (JBHI)**, 2026.
>
> 📄 **Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11520633)

## Overview

This project runs test-agnostic federated learning experiments on heterogeneous and long-tailed data.

GTAFL synthesizes federated features and trains three expert classifiers covering different long-tailed
distributions (forward / uniform / backward), combines them through ensemble aggregation, and applies
SimCLR self-supervised fine-tuning on the unlabeled test data of the target distribution. This makes the
model robust across the full spectrum of test distributions without knowing the target distribution at
training time.

## Dependencies

- Python 3.9.18 (Anaconda)
- PyTorch 2.2.1
- torchvision 0.17.0
- CUDA 12.2
- cuDNN 8.9.2.26
- numpy, tqdm, matplotlib, Pillow, tensorboard

Also verified on PyTorch 1.11.0 / torchvision 0.12.0 with an RTX 4090.

## Dataset

- CIFAR-10

It is downloaded automatically into `data/CIFAR10/` on the first run. The long-tailed class split and the
Dirichlet client partition are generated at runtime from `--seed`, `--imb_factor` and `--non_iid_alpha`.

## Parameters

The experiment parameters are controlled via the `./options.py` file:

| Argument | Default | Description |
|---|---|---|
| `num_classes` | 10 | Number of classes |
| `num_clients` | 20 | Number of all clients |
| `num_online_clients` | 8 | Number of participating local clients |
| `num_rounds` | 200 | Number of communication rounds |
| `num_epochs_local_training` | 10 | Number of local epochs |
| `batch_size_local_training` | 32 | Batch size of local training |
| `match_epoch` | 100 | Number of optimizing federated features |
| `crt_epoch` | 300 | Number of re-training classifier |
| `num_of_feature` | 100 | Number of federated features per class |
| `lr_local_training` | 0.1 | Learning rate of client updating |
| `lr_feature` | 0.1 | Learning rate of federated features optimization |
| `lr_net` | 0.01 | Learning rate of classifier re-training |
| `non_iid_alpha` | 0.5 | Control the degree of heterogeneity |
| `imb_factor` | 0.02 | Control the degree of imbalance |
| `test_training_epos` | 5 | Number of ensemble training epochs |
| `lada` | 0.01 | Additional feature number from global distribution |
| `test_distribution` | backward50 | Control the test distribution type |
| `global_sample` | off | Control the on and off of global sampling |
| `simclr` | off | Control the on and off of SimCLR |
| `vote` | **on** | Ensemble by majority vote; turn off to use softmax-weighted logits |
| `mutil_test` | off | Simultaneously measure 11 test sets |
| `resnet_size` | 8 | Backbone depth (ResNet-8) |
| `seed` | 7 | Random seed |
| `save_model_weights` | off | Save best expert checkpoints for `adaptive_ensemble.py` |

### Notes

- `test_distribution_set = ["forward50", "forward25", "forward10", "forward5", "forward2", "uniform", "backward2", "backward5", "backward10", "backward25", "backward50"]`
  The parameter of `test_distribution` should be one of these 11 test sets.
- **When `mutil_test` is true, `simclr` should be false.** SimCLR fine-tunes the backbone on the unlabeled
  data of the *target* test distribution, so a SimCLR-adapted model is no longer valid on the other 10
  distributions. To get the full 11-distribution result for GTAFL, run it once per distribution.
- `global_sample` and `simclr` are flags — write `--simclr`, not `--simclr=True`.
- The following arguments are inherited from the CReFF codebase and are currently unused:
  `mu`, `init_belta`, `save_path`, `no_classifier_retraining`. `data_fraction` only affects a log filename.

## Usage

To run the main experiment, execute the following command:

```bash
python main.py
```

### GTAFL

```bash
python main.py --test_distribution backward50 --device cuda:0 --seed 7 \
  --num_rounds 200 --num_clients 20 --num_online_clients 8 \
  --simclr --global_sample
```

### CReFF baseline

CReFF corresponds to the `uniform` expert without GTAFL's two additions, so omit both flags and read the
uniform expert's accuracy. Because no SimCLR adaptation is involved, `--mutil_test` can be used to obtain
all 11 distributions in a single run:

```bash
python main.py --test_distribution backward50 --device cuda:0 --seed 7 \
  --num_rounds 200 --num_clients 20 --num_online_clients 8 \
  --mutil_test
```

## Reading the output

The experiment log is named after the configuration:

| `simclr` | `global_sample` | Log file |
|---|---|---|
| on | on | `GTAFL_global_sample_cifar10_seed7_IF0.02_non_iid0.5.txt` |
| on | off | `GTAFL_no_sample_cifar10_seed7_IF0.02_non_iid0.5.txt` |
| off | either | `CReff_cifar10_seed7_IF0.02_non_iid0.5.txt` |

The filename only keys off `simclr`, so always check the config header at the top of the file, which
records both flags:

```
imb_factor:0.02, non_iid:0.5,global_sample:False, simclr:False,...
```

Each run appends:

- `forward_max_acc` / `uniform_max_acc` / `backward_max_acc` — best accuracy of each expert, with the
  head/medium/tail breakdown (classes 0–2 / 3–5 / 6–9). **`uniform_max_acc` is the CReFF number.**
- `ensemble_name:... accuracy:...` — the three-expert ensemble result.

With `--mutil_test`, the 11-distribution table is written to
`Mutil_data_fraction_1_cifar10_seed7_trainIF0.02non_iid0.5_resnet8.txt`. That table is produced by the
**uniform expert alone**, not by the ensemble.

SimCLR loss curves are written to `runs/simclr_loss` for TensorBoard.

## Acknowledgement

This work is built upon **CReFF** (Classifier Re-Training with Federated Features, IJCAI 2022). Our
implementation uses the CReFF framework as its foundation, and we sincerely thank the authors for
releasing their code.

> Xinyi Shang, Yang Lu*, Gang Huang, and Hanzi Wang.
> "Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features."
> *International Joint Conference on Artificial Intelligence (IJCAI), 2022.*
>
> Code: https://github.com/shangxinyi/CReFF-FL

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026gtafl,
  title={GTAFL: Addressing Test-Agnostic Long-Tailed Federated Learning in Medical Image},
  author={Chen, Guangyu and Zeng, Jingyun and Jiang, Hui},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  publisher={IEEE}
}
```
