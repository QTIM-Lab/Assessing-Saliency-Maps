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

## Models

The models used for all experiments are available [here](https://www.dropbox.com/home/Assessing-Saliency-Maps). They include 3 replicates of the InceptionV3 network trained on the RSNA Pneumonia dataset and 3 replicates trained on the SIIM-ACR Pneumothorax datasets. The splits used for the training are highlighted in [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/blob/master/pneumonia_splits.csv) and [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/blob/master/pneumonia_splits.csv) respectively. 

For the cascading randomization and repeatability/reproducibility tests, saliency map performance was evaluated on a randomly chosen sample of 100 images from the respective test sets. These images are included in both PNG and NPY form [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/tree/master/pneumonia_samples) and [here](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/tree/master/pneumothorax_samples)
