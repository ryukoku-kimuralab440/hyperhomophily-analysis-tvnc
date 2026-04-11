# Higher-Order Homophily Analysis for Hypergraphs with Time-Varying Node Categories

## Overview
This repository contains the source code used in the following paper:

> Masahito Kumano, Koki Nishimura, and Masahiro Kimura, "Higher-Order Homophily Analysis for
Hypergraphs with Time-Varying Node Categories", Applied Network Science in press, 2026.
> DOI: https://doi.org/10.xxxx/xxxxxx

## Requirements
- Python 3.13
- numpy
- scipy

## Repository Structure
```
├── README.md
├── LICENSE
├── src/
│   ├── main.py
└── sample_data/
    ├── poi_k5.pickle
    └── user_hyperedges_k5.pickle
```
## Usage
```bash
python main.py
```

## Sample Data
The sample_data/ directory contains sample data for demonstrating how to run the program. The actual datasets used in the experiments are not included in this repository. The Cookpad dataset is available from Cookpad Inc. via IDR Dataset Service of National Institute of Informatics (https://www.nii.ac.jp/dsc/idr/cookpad), subject to licensing restrictions. The Foursquare dataset is available at https://drive.google.com/file/d/1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8/view (Yang 2020). For further details, please refer to the Data Availability Statement in the paper.

## Citation
If you use this code or data, please cite:
<img width="848" height="939" alt="image" src="https://github.com/user-attachments/assets/de06f56c-b758-4324-baf0-7a709b504fd1" />
