# Tokenized Time-Series in Satellite Image Segmentation with Transformer Network for Active Fire Detection

The spatial models are from repo: https://github.com/yingkaisha/keras-unet-collection

The original Transformer implementation are from repo: https://github.com/faustomorales/vit-keras

The Dataset Downloader is from repo: https://github.com/zhaoyutim/LowResSatellitesService

Link to the paper: [DOI:10.1109/TGRS.2023.3287498](https://ieeexplore.ieee.org/document/10155171)

## Repo Structure
    .
    ├── analysis                                   # Analysis file for VIIRS Band I4 Data
    ├── data_processor                             # Data processor from VIIRS Geotiff to time-series and tokenized time-series 
    │   ├── main_dataset.py                        # Entry to generate time-series from VIIRS Geotiff
    │   ├── PreprocessingService.py                # Utils to read and write VIIRS Geotiff
    │   ├── TokenizeProcessor.py                   # Google Earth Engine Client to download and process GEDI images
    ├── deployment                                 # Essential steps for deployment to large regions
    ├── spatial_models                             # All the spatial models (ConvNet-based or Transformer-based) needed for the project 
    ├── temporal_models                            # All the temporal models (RNN and Transformer-based) needed for the project
    ├── run_cnn_model.py                           # Main Entry to run spatial models
    ├── run_seq_model.py                           # Main Entry to run temporal models
    └── README.md

## Abstract

The Visible Infrared Imaging Radiometer Suite
(VIIRS) onboard the Suomi National Polar-orbiting Partnership
(S-NPP) satellite has been used for the early detection and daily
monitoring of active wildfires. How to effectively segment the ac-
tive fire pixels from VIIRS image time-series in a reliable manner
remains a challenge because of the low precision associated with
high recall using automatic methods. For active fire detection,
multi-criteria thresholding is often applied to both low-resolution
and mid-resolution Earth observation images. Deep learning
approaches based on Convolutional Neural Networks are also
well-studied on mid-resolution images. However, ConvNet-based
approaches have poor performance on low-resolution images
because of the coarse spatial features. On the other hand, the high
temporal resolution of VIIRS images highlights the potential of
using sequential models for active fire detection. Transformer
networks, a recent deep learning architecture based on self-
attention, offer hope as they have shown strong performance
on image segmentation and sequential modelling tasks within
computer vision. In this research, we propose a Transformer-
based solution to segment active fire pixels from the VIIRS
time-series. The solution feeds a time-series of tokenized pixels
into a Transformer network to identify active fire pixels at each
timestamp and achieves a significantly higher F1-Score than prior
approaches for active fires within the study areas in California,
New Mexico, and Oregon in the US, and in British Columbia
and Alberta in Canada, as well as in Australia and Sweden.

## Methodology

![Alt text](figures/method.svg?raw=true "Methodology")

The Transformer model architecture is shown in figure above,
it closely follows the design of the original encoder of the
original Transformer and the Vision Transformer.
The major difference to the ViT paper is that our input to the
Transformer is a sequence of vectors representing one spatial
location over time, as opposed to a sequence of image patches
representing the different positions in one image. The pipeline
for active fire classification has three high-level steps
* Tokenize the sequence of images into W H time-series,
where each time-series is a sequence of tokens repre-
senting one spatial location over time and each token is
combined with an embedding of its temporal position in
the time-series.
* Encode each time-series with a masked Transformer
encoder and classify the pixel location at each time stamp
with a layer of MLP as active fire or not.
* Obtain a final sequence of predictions for the original
image sequence of active fire or not

## Results
![Alt text](figures/currowan_fire.png?raw=true "Title")

The figure above provides a showcase in Currowan Fire, Sydney happend in 2019.

## Author

#### Yu Zhao (zhao2@kth.se), Yifang Ban (yifang@kth.se), Josephine Sullivan, KTH Royal Institute of Technology, Stockholm, Sweden

## Acknowledgement
#### The research is part of the project ‘Sentinel4Wildfire’ funded by Formas, the Swedish research council for sustainable development and the project ‘EO-AI4Global Change’ funded by Digital Futures.
