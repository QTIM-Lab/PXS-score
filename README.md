# PXS-score

Chest radiographs (CXRs) are frequently acquired in patients with COVID-19. While the sensitivity and specificity of radiographs for COVID-19 is poor, CXRs can help evaluate the severity of lung disease and its change over time, but there is substantial inter-rater variability. We develop a convolutional Siamese neural network-based algorithm can calculate a continuous radiographic pulmonary disease severity score in patients with COVID-19 (Pulmonary X-ray Severity (PXS) score), which can potentially be used for longitudinal evaluation and clinical risk stratification.

Training the convolutional siamese neural network is performed in two steps: (1) pre-training on CheXpert CXRs using weak labels using contrastive loss and (2) training on CXRs from patients with COVID-19 with disease severity labels using mean square error loss.

This work is currently in submission for peer-reviewed publication. It builds on work published in npj Digital Medicine entitled "Siamese neural networks for continuous disease severity evaluation and change detection in medical imaging" (https://doi.org/10.1038/s41746-020-0255-1).

Please refer to the manuscript methodology for details. 

**Requirements**: 

- Python 3.6.9
- PyTorch 1.5.0
- CUDA 10.2
- GPU support

**How to Run**:

The working directory should contain the following Python scripts:
- PXS_preprocessing.py (DICOM pre-processing for inputs, same as applied in the CheXpert database)
- PXS_train.py (script for pre-training with CheXpert)
- PXS_train_tuning.py (script for training with CXRs from patients with COVID-19)
- PXS_classes.py (Python helper classes)
- PXS_test.py (script for testing performance)

Run the scripts in an interactive shell, like IPython. 

The final trained Siamese neural network model can be downloaded here: 

TBD...

**Citation**:

If you use this code in your work, please cite: 

TBD...

**Acknowledgments**:

The Center for Clinical Data Science (CCDS) at Massachusetts General Hospital and the Brigham and Woman's Hospital provided technical and hardware support including access to graphics processing units. The basis of the siamese neural network implementation in PyTorch is also indebted to code shared on GitHub by Harshvardhan Gupta (https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch). We thank the makers of the CheXpert data set, and for Jeremy Irvin for sharing his DICOM pre-processing code for the CheXpert data set. 

Questions? Contact us at qtimlab@gmail.com.



