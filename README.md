
### 12-Lead Imbalanced ECG Beat Classification Using Time Series ResNet

This repository contains supplementary materials from a research project for classifying ECG beat segments into diagnostic classes defined by PhysioBank.
Thus far, the following materials have been uploaded.
1. Full 12 lead ECG segments of an RBBB beat from the original ECG data set. [Here](supplemental_plots/Figure2)
2. Full 12 lead ECG segments of an RBBB beat randomly altered during ECG data augmentation. [Here](supplemental_plots/Figure3)
3. Test results in the four augmentation scenarios (EXP 1 - 4). [Here](test_results.md)
4. Detailed_results.xlsx: details of the training accuracy and test accuracy in ECG segment classification. [Here](result_details.xlsx)
5. ECG augmentation code: [README](augmentation/README.md)
  ![AugTest](imgs/rbbb.png)
6. Model code: [resnet.py](resnet/resnet.py)
  ![ResNetArch](imgs/resnet.png)
7. Class activation mapping code: [Here](class_activation_map/cam.py)
