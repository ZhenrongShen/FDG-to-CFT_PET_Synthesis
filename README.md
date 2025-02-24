# FDG-to-CFT-PET-Synthesis
Official Pytorch Implementation of [Cross-modality PET Image Synthesis for Parkinson's Disease Diagnosis: A Leap from \[<sup>18</sup>F\]FDG to \[<sup>11</sup>C\]CFT](https://link.springer.com/article/10.1007/s00259-025-07096-3) (Accepted by _European Journal of Nuclear Medicine and Molecular Imaging_). 

### Method
![Overview of Method](/figures/overview.png "Overview of Method")

We propose a two-stage framework that synthesizes \[<sup>11</sup>C\]CFT PET images from real \[<sup>18</sup>F\]FDG PET scans for subsequent automatic Parkinson's Disease (PD) diagnosis, which was based on the correlation between dopaminergic deficiency in the striatum and increased glucose metabolism in PD patients. Please refer to [the paper](https://link.springer.com/article/10.1007/s00259-025-07096-3) for more details.


### Qualitative Results
![Qualitative Results](/figures/results.png "Qualitative Results")

### Environment
- Python 3.10.10
- Pytorch 2.0.0+cu117
- monai 1.3.0
- SimpleITK, opencv-python, scikit-image, scikit-learn, numpy, matplotlib, pandas, openpyxl, tqdm


## Usage
### Training
- Refer to `configs/syn_config.yaml` and `configs/cls_config.yaml` for customizing your own configuration files. All the arguments are self-explanatory by their names and comments.

- Set the argument `DATAROOT` in `configs/{config_name}.yaml` to the dataset root. The directory structure of `DATAROOT` should be prepared as in the following example: 

```
DATAROOT
├─ PD_paired_FDG+CFT
|  ├─ {patient_ID}
|  |  ├─ {patient_ID}_FDG.nii  
|  |  └─ {patient_ID}_CFT.nii  
|  └─ ......
├─ NC_paired_FDG+CFT
|  └─ ......
├─ NC_unpaired_FDG+CFT
|  ├─ FDG
|  |  ├─ {patient_ID}_FDG.nii  
|  |  └─ ......
|  └─ CFT
|     ├─ {patient_ID}_CFT.nii  
|     └─ ......
├─ templates
|  ├─ {roi}.nii
|  └─ ......
├─ train_list.txt
├─ test_list.txt
└─ unpaired_NC_CFT_data_list
```

- The TXT files `train_list.txt` and `test_list.txt` should contain one image path (if unpaired, only read \[<sup>18</sup>F\]FDG PET image) per line as in the following example:

```
PD_paired_FDG+CFT/{patient_ID}
NC_paired_FDG+CFT/{patient_ID}
NC_unpaired_FDG+CFT/FDG/{patient_ID}_FDG.nii
......
```

- The TXT files `unpaired_NC_CFT_data_list.txt` should contain one \[<sup>11</sup>C\]CFT PET image path per line as in the following example:

```
NC_unpaired_FDG+CFT/CFT/{patient_ID}_CFT.nii
......
```

- After finishing data preparation, use the following command to train \[<sup>11</sup>C\]CFT Generator:

```
python train.py --config=syn_config
```

- After finishing the training of \[<sup>11</sup>C\]CFT Generator, set the argument `GEN_PATH` in `configs/cls_config.yaml` to the desired generator weight and use the following command to train PD Classifier:

```
python train.py --config=cls_config
```

### Testing
- Edit the testing arguments in `configs/syn_config.yaml` and use the following command to test \[<sup>11</sup>C\]CFT Generator:

```
python test.py --config=syn_config
```

- Edit the testing arguments in `configs/cls_config.yaml` and use the following command to test PD Classifier:

```
python test.py --config=cls_config
```

## :books: Citation

If you find this repository helpful, please consider citing:
```
@article{shen2025cross,
  title={Cross-modality PET image synthesis for Parkinson’s Disease diagnosis: a leap from [18F] FDG to [11C] CFT},
  author={Shen, Zhenrong and Wang, Jing and Huang, Haolin and Lu, Jiaying and Ge, Jingjie and Xiong, Honglin and Wu, Ping and Ju, Zizhao and Lin, Huamei and Zhu, Yuhua and others},
  journal={European Journal of Nuclear Medicine and Molecular Imaging},
  pages={1--10},
  year={2025},
  publisher={Springer}
}
```
