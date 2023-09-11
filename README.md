# Improved Segmentation Under Extreme Imbalance Towards Full Background Images

This is the oficial repository for the Improved Segmentation Under Extreme Imbalance Towards Full Background Images paper

## Training new models

In order to train a new model or reproduce the work described in the paper, you must first edit `Experiments/BaseConfig.py` and set the path for the datasets in your machine. Then, you can proceed by running:

```python train.py <exp_name> <dataset_name>```
 
 where `<dataset_name>` is one of the datasets previously set in `Experiments/BaseConfig.py` and `<exp_name>` is one of following:

|         **exp_name**        |                **task**                |   **encoder**   |      **decoder**     | **sampling method** |     **loss**     |                                                     **description**                                                    |
|:---------------------------:|:--------------------------------------:|:---------------:|:--------------------:|:-------------------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------:|
|    **r50_classification**   |             classification             |     resnet50    |           -          |    undersampling    |        BCE       | First stage of the two-stage pipeline. Classification model trained to predict where image is background or foreground |
|  **unet_r50_segmentation**  | segmentation only on foreground images |     resnet50    |         u-net        |    undersampling    |        BCE       | Second stage of the two-stage pipeline. Segmentation model trained only on foreground images                           |
|         **unet_r50**        |       segmentation on all images       |     resnet50    |         u-net        |    undersampling    |        BCE       | Baseline single-stage segmentation model trained on all images                                                         |
|       **unet_hc_r50**       |       segmentation on all images       |     resnet50    | u-net + hypercolumns |    undersampling    |        BCE       | Baseline single-stage segmentation model with inverted hypercolumns                                                    |
|      **isuei_unet_r50**     |       segmentation on all images       |     resnet50    |   u-net + proposed   |    undersampling    |  BCE + proposed  | Baseline single-stage segmentation with the proposed modifications                                                     |
|        **effdet_d2**        |       segmentation on all images       | efficientnet-b2 |         biFPN        |    undersampling    |        BCE       | Single-stage segmentation model based on EfficientDet                                                                  |
|         **isuei_d2**        |       segmentation on all images       | efficientnet-b2 |   biFPN + proposed   |    undersampling    |  BCE + proposed  | Single-stage segmentation model based on EfficientDet with the proposed modifications                                  |
|         **fapn_r50**        |       segmentation on all images       |     resnet50    |         fapn         |    undersampling    |        BCE       | Single-stage segmentation model with FaPNet decoder                                                                    |
|      **isuei_fapn_r50**     |       segmentation on all images       |     resnet50    |         fapn         |    undersampling    |  BCE + proposed  | Single-stage segmentation model with FaPNet decoder and the proposed modifications                                     |
|      **unet_r50_focal**     |       segmentation on all images       |     resnet50    |         u-net        |    undersampling    |       focal      | Baseline single-stage segmentation model using Focal loss instead of BCE                                               |
|   **isuei_unet_r50_focal**  |       segmentation on all images       |     resnet50    |   u-net + proposed   |    undersampling    | focal + proposed | Baseline single-stage segmentation model with the proposed modifications using Focal loss instead of BCE               |
| **unet_r50_randomsampling** |       segmentation on all images       |     resnet50    |         u-net        |        random       |        BCE       | Baseline random sampling experiment                                                                                    |
|  **unet_r50_oversampling**  |       segmentation on all images       |     resnet50    |         u-net        |     oversampling    |        BCE       | Oversampling technique to tackle the imbalance                                                                         |
|  **unet_r50_undersampling** |       segmentation on all images       |     resnet50    |         u-net        |    undersampling    |        BCE       | Undersampling technique to tackle the imbalance                                                                        |

## Validating the trained models
Once the model is trained, you can evaluate it.

### Single-stage evaluation
Segmentation models trained on all data (background + foreground) can be evaluated directly as a single-stage approach. In order to do so please run:

```python single_stage_eval.py <exp_name>  <dataset_name> <model_weights_path>```

### Two-stage evaluation
In order to evaluate the two-stage pipeline, you need to have both a classification and segmentation (either trained on all images or only on foreground) models.

```python two_stage_eval.py <dataset_name> <cls_exp_name> <cls_weights_path> <seg_exp_name> <seg_weights_path>```

## Roadmap/Future improvements
- [x] Clean-up and refactor the code
- [x] Update `torch` version
- [x] Replace NVIDIA-AMP package in favor for `torch.amp`
- [ ] Create `Dockerfile` with requirements, ready for training and validation
- [ ] Upload trained weights
- [ ] Futher improve code