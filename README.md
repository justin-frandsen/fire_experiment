# fire_experiment

This is my code for an experiment analyzing how primates attend to fire. I utilized a saliency model GBVS to assess if attending to fire was accounted for via saliency or through a quality of the fire itself.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/justin-frandsen/fire_experiment.svg)](https://github.com/justin-frandsen/fire_experiment)
[![Last Commit](https://img.shields.io/github/last-commit/justin-frandsen/fire_experiment.svg)](https://github.com/justin-frandsen/fire_experiment/commits/main)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction
This project aims to understand the visual attention of primates towards fire. By using the Graph-Based Visual Saliency (GBVS) model, we can determine whether the attention is due to the inherent saliency of fire or other factors.

## Installation
To run this project, you need to have Python installed. You can install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
Additionally, in a separate repository you need to clone GBVS using:
```bash
git clone https://github.com/shreelock/gbvs
```
once cloned copy the saliency_models folder into the fire_experiment folder using:
```bash
cp -r gbvs/saliency_models fire_experiment/
```

## Usage

Note: This code was run on **Python 3.13.6**. The `requirements.txt` is set up with compatible package versions. Users running other Python versions may need to adjust package versions accordingly.

To use the code, run the following command:
```bash
python roi_saliency_extractor.py
```

## Results
The results of the experiment will be saved in the `results` directory. You can analyze these results to understand the attention patterns.

## References
- [GBVS on GitHub](https://github.com/shreelock/gbvs)
- Original Paper on GBVS: Harel, J., Koch, C., & Perona, P. (2006). Graph-based visual saliency. Advances in neural information processing systems, 19.



