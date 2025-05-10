# Pattern-aware LLM-based Online Time Series Forecasting

## Introduction
Implementation of **Pattern-aware LLM-based Online Time Series Forecasting**.  
This method for online time series forecasting in practical scenarios leverages frequency decomposition to represent the time series as a combination of frequency bases, with learned knowledge for each basis. It integrates a pre-trained LLM with the time series backbone to enable effective adaptation to streaming data in data-scarce online settings, utilizing the LLM’s extensive knowledge.

## Basics
1. `run_main.py` : Execute the entire experiment.
2. `models/LLM4OTSF.py` : The structure of the proposed model.
3. `utils/tools.py` : The operation of the training phase and online phase
4. `data_provider/data_loader.py` : Load the data to be used in the training phase and online phase based on the dataset, prediction length, split ratio, and other factors.


## Dataset
You can download the datasets (ETT, ECL, Weather, Traffic) from the following links.
1. ETT
  https://github.com/zhouhaoyi/ETDataset
2. Weather
  https://www.ncei.noaa.gov/data/local-climatological-data/
3. ECL
  https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
4. Traffic
  https://pems.dot.ca.gov/


## Requirments
- Pytorch version: 3.11
- torch: 2.2.2
- accelaerate: 0.28.0
- einops: 0.7.0
- matplotlib: 3.7.0
- numpy: 1.23.5
- pandas: 1.5.3
- scikit_learn: 1.2.2
- scipy: 1.12.0
- transformers: 4.31.0
- deepspeed: 0.14.0


## Arguments
- `--data_path:` Path of the dataset.<br>
	- usage example :`--data_path ETTh2.csv`
- `--model:` Types of models used for training.<br>
	- usage example :`--model llm4otsf`
- `--seq_len:` Input sequence length.<br>
	- usage example : `--seq_len 96`
- `--pred_len:`  prediction sequence length.<br>
	- usage example :`--pred_len 24`
- `--llm_model:` Types of LLMs to be used alongside the time series backbone.<br>
	- usage example :`--llm_model LLAMA`
- `--learning_rate:` Learning rate.<br>
	- usage example : `--learning_rate 0.0001`
- `--wavelet_level:` A hyperparameter that controls the level of Discrete Wavelet Transform.<br>
	- usage example : `--wavelet_level 2`
- `--train_epochs:` Epochs of the training phase.<br>
	- usage example : `--train_epochs 10`
- `--batch_size:` Batch size of the training phase.<br>
	- usage example : `--batch_size 16`



## Source code of the backbone network
- The source code of the backbone network is referenced from:
  - https://github.com/salesforce/fsnet/
  - https://github.com/yfzhang114/OneNet
  - https://github.com/yyalau/iclr2025_dsof
