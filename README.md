# SROSDA (ICCV 2021)
implementation of the ICCV work "SR-OSDA". 

## Data
---
- [I2AwA](./data/I2AwA/dataset_info.txt)
- [N2AwA](./data/N2AwA/classes.txt)

To extract pre-trained ResNet-50 features, check script:

```shell
./data/N2AwA/features/extract_resnet_features.ipynb
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

### Step 2:

### Step 3:

## Evaluation
---

- Open-set Domain Adaptation Task

> $OS^*$: 
>
> $OS^\diamond$: 
>
> $OS$: 

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
