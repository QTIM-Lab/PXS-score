# PXS-score

Chest radiographs (CXRs) are frequently acquired in patients with COVID-19. While the sensitivity and specificity of radiographs for COVID-19 is poor, CXRs can help evaluate the severity of lung disease and its change over time, but there is substantial inter-rater variability. We develop a convolutional Siamese neural network-based algorithm can calculate a continuous radiographic pulmonary disease severity score in patients with COVID-19 (Pulmonary X-ray Severity (PXS) score), which can potentially be used for longitudinal evaluation and clinical risk stratification.

Training the convolutional siamese neural network is performed in two steps: (1) pre-training on CheXpert CXRs using weak labels using contrastive loss and (2) training on CXRs from patients with COVID-19 with disease severity labels using mean square error loss.

The original work was published as "Automated Assessment and Tracking of COVID-19 Pulmonary Disease Severity on Chest Radiographs using Convolutional Siamese Neural Networks" in Radiology: Artificial Intelligence (Li et al. 2020, https://pubs.rsna.org/doi/10.1148/ryai.2020200079). It builds on work published in npj Digital Medicine entitled "Siamese neural networks for continuous disease severity evaluation and change detection in medical imaging" (https://doi.org/10.1038/s41746-020-0255-1). 

Since then, an updated model with an additional model tuning step analogous to step 2 of training above (mean square error loss training using disease severity labels) has been trained, which has shown improved performance (i.e. correlation with manual radiologist labels). The manuscript of this model will be submitted as a pre-print and for peer review. 

To differentiate these two models, the original model published in Radiology: Artificial Intelligence is PXS-original and the updated model is PXS-updated.

Please refer to the manuscript methodologies for details. 

**Requirements**: 

- Python 3.6.9
- PyTorch 1.5.0
- CUDA 10.2
- GPU support

**How to Run**:

The working directory should contain the following Python scripts:

- PXS_classes.py (Python helper classes)

(1) - for pre-training on CheXpert data
- PXS_label_processor.py (for processing CheXpert labels for input into PXS_train.py)
- PXS_train.py (script for pre-training with CheXpert)

(2) - for training on COVID-19 data
- PXS_preprocessing.py (DICOM pre-processing for inputs, same as applied in the CheXpert database)
- PXS_train_tuning.py (script for training with CXRs from patients with COVID-19)
    - in PXS-updated, this script is used again to tune the model on a new dataset

(3) - for testing on COVID-19 data
- PXS_test.py (script for testing performance used in PXS-original)
- PXS_test_direct_from_DICOM.py (script for testing performance used in PXS-updated, which includes comparison to a different set of anchor images and incorporates an automatic cropping step before the model inference)

Run the scripts in an interactive shell, like IPython. 

**Citation**:

If you use this code in your work, please cite: 

Matthew D. Li, Nishanth Thumbavanam Arun, Mishka Gidwani, Ken Chang, Francis Deng, Brent P. Little, Dexter P. Mendoza, Min Lang, Susanna I. Lee, Aileen O’Shea, Anushri Parakh, Praveer Singh, and Jayashree Kalpathy-Cramer. Automated Assessment and Tracking of COVID-19 Pulmonary Disease Severity on Chest Radiographs using Convolutional Siamese Neural Networks. Radiology: Artificial Intelligence 2020 2:4. (https://pubs.rsna.org/doi/10.1148/ryai.2020200079)

Updated model manuscript pending.

**Acknowledgments**:

The Center for Clinical Data Science (CCDS) at Massachusetts General Hospital and the Brigham and Woman's Hospital provided technical and hardware support including access to graphics processing units. The basis of the siamese neural network implementation in PyTorch is also indebted to code shared on GitHub by Harshvardhan Gupta (https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch). We thank the makers of the CheXpert data set, and for Jeremy Irvin for sharing his DICOM pre-processing code for the CheXpert data set. 

Questions? Contact us at qtimlab@gmail.com.



