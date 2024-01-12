# Feature-decoupling-and-weighted-loss

## Description
This repository contains train code for applying the feature decoupling cRT, weighted binary cross entropy loss and focal loss on the best methods that we employeed (ConvNext + ML-decoder + data augmetation).

## Run
1. Use the same environment as the ML-decoder repository.
2. Download and put all the data named MICCAI_\*.tfrecords and MICCAI_\*.tfindex from [here](https://drive.google.com/drive/folders/1vIGUboqMDf4osIzKLp0AF0ow1kgLT70x?usp=sharing) under the data folder
3. Download and put the content under the classBalance folder from [here](https://drive.google.com/drive/folders/1vIGUboqMDf4osIzKLp0AF0ow1kgLT70x?usp=sharing) under the data/classBalance folder
4. Download and put the best.pt file from [here](https://drive.google.com/drive/folders/1Bo8SI5RyqC36ih-qXx8uBmzSUAxx8PZm?usp=sharing) under the weight folder
5. Run featureDecoupling.py | focalLoss.py | weightedLoss.py to reproduce our results
