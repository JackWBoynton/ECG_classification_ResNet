
### Data Augmentation for 12-Lead Imbalanced ECG Beat Classification Using Time Series ResNet

This repository contains supplementary materials from a research project for classifying ECG beat segments into diagnostic classes defined by PhysioBank.
Thus far, the following materials have been uploaded.
1. A paper published at SN Computer Science ([pdf](Published_SNCS_Nov2021.pdf))
2. Full 12 lead ECG segments of an RBBB beat from the original ECG data set ([link](supplemental_plots/Figure2)).
3. Full 12 lead ECG segments of an RBBB beat randomly altered during ECG data augmentation ([link](supplemental_plots/Figure3)).
4. Test results in the four augmentation scenarios (EXP 1 - 4) ([link](test_results.md)).
5. Detailed_results.xlsx: details of the training accuracy and test accuracy in ECG segment classification ([link](result_details.xlsx)).
6. ECG filtering and segmentation ([README](preproc)).
7. ECG augmentation ([README](augmentation/README.md)).<br/>
   Sample segment:<br/>
  ![AugTest](imgs/rbbb.png)
7. ResNet model ([link](resnet)).<br/>
   ResNet architecture:<br/>
  ![ResNetArch](imgs/resnet.png)
8. Class activation mapping ([link](class_activation_map)).
