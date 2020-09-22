### 12-Lead Imbalanced ECG Beat Classification Using Time Series ResNet

This repository contains supplementary materials from a research project for classifying ECG beat segments into diagnostic classes defined by PhysioBank.
Thus far, the following materials have been uploaded.
1. Full 12 lead ECG segments of an RBBB beat from the original ECG data set. [here](supplemental_plots/Figure2)
2. Full 12 lead ECG segments of an RBBB beat randomly altered during ECG data augmentation. [here](supplemental_plots/Figure3)
3. Detailed_results.xlsx: details of the training accuracy and test accuracy in ECG segment classification.
4. ECG augmentation code: [README](https://github.com/jackwboynton/ecg-augmentation/README.md)
  ![AugTest](imgs/rbbb.png)
5. Model code: [resnet.py](resnet/resnet.py)
  ![ResNetArch](imgs/resnet.png)

TODO:
1. Add CAM code
