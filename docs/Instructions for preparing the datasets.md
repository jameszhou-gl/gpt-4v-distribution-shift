

# Instructions for preparing the datasets

[TOC]

## Download

Most of the datasets, including PACS, OfficeHome, Terra Incognita, and WILDSCamelyon, can be downloaded with [this script](https://github.com/m-Just/DomainBed/blob/main/domainbed/scripts/download.py).
Other datasets are also publicly available but need to be downloaded manually.



## Directory structure

Make sure that the directory structure of each dataset is arranged as follows:

### Datasets in [Domainbed](https://arxiv.org/pdf/2007.01434.pdf)

#### PACS

```bash
PACS
├── art_painting
  ├── dog
  ├── elephant
  ├── ...
├── cartoon
├── photo
└── sketch
```

#### VLCS

```bash
VLCS
├── Caltech101
  ├── bird
  ├── car
  ├── ...
├── LabelMe
├── SUN09
└── VOC2007
```

#### OfficeHome

```bash
office_home
├── Art
  ├── Alarm_Clock
  ├── Backpack
  ├── ...
├── Clipart
├── Product
└── Real World
```

#### Terra Incognita

```bash
terra_incognita
├── location_38
  ├── bird
  ├── bobcat
  ├── ...
├── location_43
├── location_46
└── location_100
```

#### Camelyon17-WILDS

```bash
camelyon17_v1.0
├── patches
└── metadata.csv
```

#### DomainNet

```bash
domain_net
├── clipart
├── infograph
├── painting
├── quickdraw
├── real
└── sketch
```

#### COVID

```bash
domain_net
├── source
  ├── normal
  ├── pneumonia
├── target
  ├── normal
  ├── pneumonia
  ├── COVID19
```

#### DrugOOD_assay
run To_image_assay.py with drugood_assay.txt to get DrugOOD_assay
```bash
DrugOOD_assay
├── domain01
  ├── inactive
  ├── active
├── ...
└── domain80
```
#### DrugOOD_scaffold

```bash
DrugOOD_scaffold
├── domain01
  ├── inactive
  ├── active
├── ...
└── domain12542
```

#### PACS_gaussion

```bash
PACS_gaussion
├── art_painting
  ├── dog
  ├── elephant
  ├── ...
├── cartoon
├── photo
└── sketch
```

#### PACS_unseen

```bash
PACS_unseen
├── art_painting
  ├── dog
  ├── elephant
  ├── ...
├── cartoon
├── photo
└── sketch
```


### Datasets in [CLIP](https://arxiv.org/pdf/2103.00020.pdf)

#### ImageNet-V2

```bash
# download from https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main
imagenetv2-matched-frequency-format-val
├── 118
└── ...
```



#### ImageNet-R

```bash
# official website: https://github.com/hendrycks/imagenet-r
# download: https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
imagenet-r
├── n01616318
└── ...
```



#### ImageNet Sketch

```bash
# download from https://github.com/HaohanWang/ImageNet-Sketch
imagenet-sketch
├── n01498041
└── ...
```



#### ImageNet-A

```bash
# download from https://github.com/hendrycks/natural-adv-examples
imagenet-a
├── n01498041
└── ...
```



#### ObjectNet

```bash

```



----

### Todo

#### CelebA

```
celeba
├── img_align_celeba
└── blond_split
    ├── tr_env1_df.csv
    ├── tr_env2_df.csv
    └── te_env_df.csv
```

#### NICO

```
NICO
├── animal
├── vehicle
└── mixed_split_corrected
    ├── env_train1.csv
    ├── env_train2.csv
    ├── env_val.csv
    └── env_test.csv
```



> Reference:
>
> [OoD-Bench](https://github.com/m-Just/OoD-Bench/tree/main), [DomainBed](https://github.com/facebookresearch/DomainBed/tree/main), 
