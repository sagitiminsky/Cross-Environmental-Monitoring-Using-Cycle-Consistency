PYTHONPATH should be: CellEnMon-Research

Suggested presets:
+ download pycharm
+ install Ansible Vault plugin


**WandB**
https://wandb.ai/sagitiminsky/CellEnMon_CycleGan

**Documentation:** -
Please look in the secrets.yaml required creds. You can find the master password in our cellenmon whatsapp
description

https://app.swimm.io/workspaces/jllXRmECUZBMawQBnXtN

**Medium:**
https://medium.com/me/stories/public

# PyNNcml
A python toolbox based on PyTorch which utilized neural network for rain estimation and classification from commercial microwave link (CMLs) data. This toolbox provides an implementation of algorithms for extracting rain-rate using neural networks and CMLs. Addinaly this project provides an example dataset with data from two CMLs and implementation of performance and robustness metrics.  
# Install
Installation via pip:
```
pip install pynncml

```


# Projects Structure

1. Wet Dry Classification
2. Baseline 
3. Power Law 
4. Rain estimation
5. Metrics
6. Robustness

# Dataset
The collection of datasets that were used for training the model can be found at: [Datasets](s3://cellenmon)
Please note that the dataset are not publicly available. To gain access to the bucket please open an issue.

# Usage
The following examples:



# Model Zoo
In this project we supply a set of trained networks in our [Model Zoo], those networks are trained on our own dataset which is not publicly available.

# Contributing

If you find a bug or have a question, please create a GitHub issue.



# Publications

Please cite one of following paper if you found our neural network model useful. Thanks!
>[1] Timinsky, Sergey Timinsky, J. Ostrometzky and H. Messer ""
```
@inproceedings{habi2018wet,
  title={Wet-Dry Classification Using LSTM and Commercial Microwave Links},
  author={Habi, Hai Victor and Messer, Hagit},
  booktitle={2018 IEEE 10th Sensor Array and Multichannel Signal Processing Workshop (SAM)},
  pages={149--153},
  year={2018},
  organization={IEEE}
} 
```

Also this package contains the is based on the following papers:

>[2] Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"

```
@inproceedings{habi2018wet,
  title={Wet-Dry Classification Using LSTM and Commercial Microwave Links},
  author={Habi, Hai Victor and Messer, Hagit},
  booktitle={2018 IEEE 10th Sensor Array and Multichannel Signal Processing Workshop (SAM)},
  pages={149--153},
  year={2018},
  organization={IEEE}
} 

```

>[3] Habi, Hai Victor and Messer, Hagit. "RNN MODELS FOR RAIN DETECTION"

```
@inproceedings{habi2019rnn,
  title={RNN MODELS FOR RAIN DETECTION},
  author={Habi, Hai Victor and Messer, Hagit},
  booktitle={2019 IEEE International Workshop on Signal Processing Systems  (SiPS)},
  year={2019},
  organization={IEEE}
} 

```

>[4] Habi, Hai Victor. "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links"

```
@article{habi2020,
  title={Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links},
  author={Habi, Hai Victor},
  journal={M.Sc. Thesis, Tel Aviv University},
  year={2019}
}

```

[5] J. Ostrometzky and H. Messer, “Dynamic determination of the baselinelevel in microwave links for rain monitoring from minimum attenuationvalues,”IEEE Journal of Selected Topics in Applied Earth Observationsand Remote Sensing, vol. 11, no. 1, pp. 24–33, Jan 2018.

[6] M. Schleiss and A. Berne, “Identification of dry and rainy periods usingtelecommunication  microwave  links,”IEEE  Geoscience  and  RemoteSensing Letters, vol. 7, no. 3, pp. 611–615, 2010

[7] Jonatan Ostrometzky, Adam Eshel, Pinhas Alpert, and Hagit Messer. Induced bias in attenuation measurements taken from commercial microwave links. In 2017 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3744–3748. IEEE,2017. <br>

[8] Jonatan Ostrometzky, Roi Raich, Adam Eshel, and Hagit Messer.
Calibration of the
attenuation-rain rate power-law parameters using measurements from commercial microwave networks. In 2016 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 3736–3740. IEEE, 2016.

If you found on of this methods usefully please cite.
