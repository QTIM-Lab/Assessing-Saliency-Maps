# Assessing Saliency Maps

([Link to article](https://www.medrxiv.org/content/10.1101/2020.07.28.20163899v1))

In this study, we comprehensively evaluate popular saliency map methods for medical imaging classification models trained on the SIIM-ACR Pneumothorax Segmentation and RSNA Pneumonia Detection datasets in terms of 4 key criteria for trustworthiness: 
1. Utility 
2. Sensitivity to weight randomization 
3. Repeatability 
4. Reproducibility 

The combination of these trustworthiness criteria provide a blueprint for us to objectively assess a saliency map's localization capabilities (localization utility), sensitivity to trained model weights (versus randomized weights), and robustness with respect to models trained with the same architectures (repeatability) and different architectures (reproducibility). These criteria are important in order for a clinician to trust the saliency map output for its ability to localize the finding of interest.

![fig1](figures/fig1.png)

For model interpretation, we evaluate the following saliency maps for their trustworthiness: Gradient Explanation (GRAD), Smoothgrad (SG), Integrated Gradients (IG), Smooth IG (SIG), GradCAM, XRAI, Guided-backprop (GBP), and Guided GradCAM (GGCAM). 

## Experiments

### Utility
We evaluate the localization utility of each saliency method by quantifying their intersection with ground truth pixel-level segmentations available from the SIIM-ACR Pneumothorax dataset and ground truth bounding boxes available from the RSNA Pneumonia dataset respectively. To capture the intersection between the saliency maps and segmentations or bounding boxes, we consider the pixels inside the segmentations to be positive labels and those outside to be negative. Each pixel of the saliency map is thus treated as an output from a binary classifier. Hence, all the pixels in the saliency map can be jointly used to compute the area under the precision-recall curve (AUPRC) utility score. 

### Sensitivity to Trained vs Random Model Weights
To investigate the sensitivity of saliency methods under changes to model parameters and identify potential correlation of particular layers to changes in the maps, we employ cascading randomization. In cascading randomization, we successively randomize the weights of the trained model beginning from the top layer to the bottom one, which results in erasing the learned weights in a gradual fashion. We use the Structural SIMilarity (SSIM) index of the original saliency map with the saliency maps generated from the model after each randomization step to assess the change of the corresponding saliency maps

### Repeatability and Reproducibility
We conduct repeatability tests on the saliency methods by comparing maps from a) different randomly initialized instances of models with the same architecture trained to convergence (intra-architecture repeatability) and b) models with different architectures each trained to convergence (inter-architecture reproducibility) using SSIM between saliency maps produced from each model. These experiments are designed to test if the saliency methods produce similar maps with a different set of trained weights and whether they are architecture agnostic (assuming that models with different trained weights or architectures have similar classification performance).

More details on the experiments can be found in the ([manuscript](https://www.medrxiv.org/content/10.1101/2020.07.28.20163899v1))

## Models

The models used for all experiments are available [here](https://www.dropbox.com/home/Assessing-Saliency-Maps). They include 3 replicates of the InceptionV3 network trained on the RSNA Pneumonia dataset and 3 replicates trained on the SIIM-ACR Pneumothorax datasets. The splits used for the training are highlighted [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/blob/master/pneumonia_splits.csv) and [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/blob/master/pneumonia_splits.csv) respectively. 

For the cascading randomization and repeatability/reproducibility tests, saliency map performance was evaluated on a randomly chosen sample of 100 images from the respective test sets. These images are included in both PNG and NPY form [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/tree/master/pneumonia_samples) and [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/tree/master/pneumothorax_samples)
