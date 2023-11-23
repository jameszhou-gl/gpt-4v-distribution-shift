

## How good is GPT-4V in distribution shifts?



1. Abstract
2. Introduction
3. Observation (takeaways)
4. Main results (by subjects: natural, protein, medical)
5. Set Clip/LLaVA as baselines
6. Multimodal in-context test (set examples from source domains as in-context); Qualitative analysis in various types of distribution shift
7. Leverage ControlNet/GPT-4V to modify the image styles
8. Calibration (based on confidence scores output by GPT-4V)



### V1 Experiments

|                           Dataset                            | Subjects |   Task Types   |   Shift Types   |
| :----------------------------------------------------------: | :------: | :------------: | :-------------: |
|                             PACS                             | Natural  | Classification | Diversity Type? |
|                             VLCS                             | Natural  | Classification | Diversity Type? |
|                          OfficeHome                          | Natural  | Classification | Diversity Type? |
|                          DomainNet                           | Natural  | Classification |                 |
|                       Terra Incognita                        | Natural  | Classification |                 |
|                           iWildCam-WILDS                     | Natural  | Classification |                 |
|                         Camelyon17-WILDS                     | Medical  | Classification |                 |
|                         COVID-19-x-ray                       | Medical  | Classification |                 |
|                          HAM10000                            | Medical  | Classification |                 |
|                         NIH-Chest-X-ray-14                   | Medical  | Classification |                 |
|                         DrugOOD-Scaffold                     | Molecular| Classification |                 |
|                         DrugOOD-Assay                        | Molecular| Classification |                 |
|                           FMoW-WILDS                         | Satellite| Classification |                 |





### V2 Experiments
|                           Dataset                            | Subjects |   Task Types   |   Shift Types   |
| :----------------------------------------------------------: | :------: | :------------: | :-------------: |
|            [ImageNetv2](https://imagenetv2.org/)             | Natural  | Classification |                 |
| [ImageNet-R](https://paperswithcode.com/dataset/imagenet-r)  | Natural  | Classification |                 |
| [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch) | Natural  | Classification |                 |
| [ImageNet-A](https://paperswithcode.com/dataset/imagenet-a)  | Natural  | Classification |                 |
 