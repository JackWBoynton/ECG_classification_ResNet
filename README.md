
### 12-Lead Imbalanced ECG Beat Classification Using Time Series ResNet

This repository contains supplementary materials from a research project for classifying ECG beat segments into diagnostic classes defined by PhysioBank.
Thus far, the following materials have been uploaded.
1. Full 12 lead ECG segments of an RBBB beat from the original ECG data set ([link](supplemental_plots/Figure2)).
2. Full 12 lead ECG segments of an RBBB beat randomly altered during ECG data augmentation ([link](supplemental_plots/Figure3)).
3. Test results in the four augmentation scenarios (EXP 1 - 4) ([link](test_results.md)).
4. Detailed_results.xlsx: details of the training accuracy and test accuracy in ECG segment classification ([link](result_details.xlsx)).
5. ECG filtering and segmentation: ([link](preproc))
6. ECG augmentation: [README](augmentation/README.md).<br/>
   Sample segment:<br/>
  ![AugTest](imgs/rbbb.png)
7. ResNet model: ([link](resnet)).<br/>
   ResNet architecture:<br/>
  ![ResNetArch](imgs/resnet.png)
8. Class activation mapping: ([link](class_activation_map)).
