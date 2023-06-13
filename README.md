# SROSDA (ICCV 2021)
implementation of the ICCV work **Towards Novel Target Discovery Through Open-Set Domain Adaptation** [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.pdf)].

:zap: ***Please check the extension journal work "Interpretable Novel Target Discovery Through Open-Set Domain Adaptation" ([XSR-OSDA](https://github.com/scottjingtt/XSROSDA)).***

## Data Preparation
---
- [N2AwA](./data/N2AwA/classes.txt): DomainNet & AwA2
- [I2AwA](./data/I2AwA/dataset_info.txt): 3D2 & AwA2

(1) To extract pre-trained ResNet-50 features, check script:

```shell
./data/N2AwA/features/extract_resnet_features.ipynb
```
(2) Collect attributes for all samples based on their labels, check script:

```shell
./data/N2AwA/attributes/check_N2AwA_data.ipynb
```

## Dependencies
---
- Python 3.6
- Pytorch 1.1


## Training
---
### Step 1: Initialization clustering on target data (Seen/Unseen Initialization)
```shell
./data/N2AwA/refine_cluster-samples.ipynb
```

*Note:* Or use our clustering initialization results `./data/N2AwA/` directly.

### Step 2: Train with the initialized clustering and pseudo labels on the extracted features.
```shell
python main.py
```

## Evaluation
---

- Open-set Domain Adaptation Task

> $OS^*$: class-wise average accuracy on the seen categories.
>
> $OS^\diamond$: class-wise average accuracy on the unseen categories correctly classified as "unknown".
>
> $OS$: $\frac{OS^* \times C_{shr} + OS^\diamond}{C_{shr} + 1}$

*$C_{shr}$ is the number of shared categories between the source and target domains.*

- Semantic-Recovery Open-Set Domain Adaptation Task

> $S$: class-wise average accuracy on shared classes
>
> $U$: class-wise average accuracy on unknown classes
>
> $H = \frac{2 \times S \times U}{ S + U}$

## Citation
---
If you think this work is interesting, please cite:
```
@InProceedings{Jing_2021_ICCV,
author = {Jing, Taotao and Liu, Hongfu and Ding, Zhengming},
title = {Towards Novel Target Discovery Through Open-Set Domain Adaptation},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2021}
}
```

## Contact
---
If you have any questions about this work, feel free to contact
- tjing@tulane.edu
