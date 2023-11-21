

## How good is GPT-4V in distribution shifts?



1. Abstract
2. Introduction
3. Observation (takeaways)
4. Main results (by subjects: natural, protein, medical)
5. Set Clip/LLaVA as baselines
6. Multimodal in-context test (set examples from source domains as in-context); Qualitative analysis in various types of distribution shift
7. Leverage ControlNet/GPT-4V to modify the image styles
8. Calibration (based on confidence scores output by GPT-4V)



#### Experiments

|                           Dataset                            | Subjects |   Task Types   |   Shift Types   |
| :----------------------------------------------------------: | :------: | :------------: | :-------------: |
|                             PACS                             | Natural  | Classification | Diversity Type? |
|                             VLCS                             | Natural  | Classification | Diversity Type? |
|                          OfficeHome                          | Natural  | Classification | Diversity Type? |
|                          DomainNet                           | Natural  | Classification |                 |
|                       Terra Incognita                        | Natural  | Classification |                 |
|                                                              |          |                |                 |
|            [ImageNetv2](https://imagenetv2.org/)             | Natural  | Classification |                 |
| [ImageNet-R](https://paperswithcode.com/dataset/imagenet-r)  | Natural  | Classification |                 |
| [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch) | Natural  | Classification |                 |
| [ImageNet-A](https://paperswithcode.com/dataset/imagenet-a)  | Natural  | Classification |                 |
|                         DrugOOD-Scaffold                     | Molecular| Classification |                 |
|                         DrugOOD-Assay                        | Molecular| Classification |                 |
|                                                              | Medical  |                |                 |
| [Camelyon17-WILDS](https://pan.baidu.com/s/1mIzSewImtEisclPtTHGSyw) | Medical  |         |                 |
| [Camelyon16](https://pan.baidu.com/s/1UW_HLXXjjw5hUvBIUYPgbA)| Medical  |                |                 |
| [TCGA](https://portal.gdc.cancer.gov/)                       | Medical  |                |                 |
|                           iWildCam-WILDS                     | Natural  | Classification |                 |
|        [RxRx1-WILDS](https://www.rxrx.ai/rxrx1)              | Biology  |                |                 |
|        [IXI](http://brain-development.org/ixi-dataset/)      | Medical  |                |                 |
|                           FMoW-WILDS                         | Natural  |                |                 |
|                           PovertyMap-WILDS                   | Natural  |                |                 |
|                           GlobalWheat-WILDS                  | Natural  |                |                 |
|                           OGB-MolPCBA-WILDS                  | Molecular|                |                 |
|[BraTS](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) | Medical  |                |                 |
|[MM-WHS](https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA)| Medical|                |                 |
|                             Todo                             | Natural  |   Regression   |                 |
|                                                              |          |                |                 |
|                                                              |          |                |                 |
|                                                              |          |                |                 |
|                                                              |          |                |                 |
|                                                              |          |                |                 |

