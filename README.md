# Intrusion Detection System (IDS) Project

## Overview
This repository contains a Python-based Intrusion Detection System (IDS) project that uses machine learning to classify network-connection records as either "Normal" or "Attack". The project trains multiple classical classifiers (including Logistic Regression — the primary model used for the real-time simulation) on the KDD-style CSV dataset and includes a small real-time simulation that streams samples and prints alerts to the console.

## Objectives
- Train and evaluate several ML classifiers on the provided dataset.
- Save trained models and preprocessing artifacts for later use.
- Provide a simple real-time simulation that loads the saved Logistic Regression model and prints per-sample alerts.

## Dataset
- `kddtrain.csv` — training data (raw KDD-style CSV, no header)
- `kddtest.csv` — test data (raw KDD-style CSV, no header)

Both files are expected to be in the project root. The code reads them with `header=None` and uses column 0 as the label and columns 1:42 as features.

## Files and Purpose
- `all.py` — training and evaluation script. Trains multiple models, evaluates on test subset, saves models, scalers, and prediction outputs under `classical/`.
- `realtime_ids.py` — a small runtime script that loads `classical/logistic_model.joblib` and `classical/normalizer.joblib` and simulates streaming classification of test samples.
- `classical/` — directory used to store saved models and output files (e.g., `logistic_model.joblib`, `normalizer.joblib`, `predictedlabel*.txt`, `predictedproba*.txt`).
- `requirements.txt` — Python package requirements for running the project.
- `README.md` — this file (project documentation).

## Important Notes about Paths and Running
- The scripts use relative paths anchored at the project root. To avoid file-not-found errors, run the scripts from the `IDS_Project` directory (the folder that contains `all.py`, `realtime_ids.py`, and the CSV files).

- `all.py` saves models and outputs into the `classical/` subdirectory. Make sure the `classical` folder exists. The repository now includes logic to create this folder if missing.

- `realtime_ids.py` loads the saved artifacts from `classical/` using `os.path.join(base_dir, 'classical', ...)`, so it expects the `classical` directory underneath the project root.

## How to Set Up and Run
1. Create a virtual environment (recommended):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
2. Install dependencies:
  ```powershell
  pip install -r requirements.txt
  ```
3. Train models & generate outputs (this will save artifacts into `classical/`):
  ```powershell
  python all.py
  ```
4. Run the real-time simulation (after step 3 finished):
  ```powershell
  python realtime_ids.py
  ```

## Output and Evaluation
- Training script prints evaluation metrics (accuracy, precision, recall, F1-score) for each classifier it trains.
- The `classical/` folder contains prediction outputs and saved artifacts that `realtime_ids.py` depends on.

## Troubleshooting
- FileNotFoundError when loading `classical/logistic_model.joblib` — ensure you have run `all.py` and that the `classical` directory exists and contains the expected `.joblib` files.
- If running from a different working directory, set `base_dir = os.path.dirname(os.path.abspath(__file__))` and use `os.path.join(base_dir, ...)` to reliably construct paths. The scripts in this repo use that approach.

## Citations
Please keep any original dataset or literature citations that may accompany the dataset or assignment materials. (This README does not remove any in-repo citations.)

## Contact / Notes
If you want, I can also:
- Add a `requirements.lock` or `pip freeze` output.
- Convert the training script to accept CLI args for dataset paths and output directory.
- Add unit tests or a small demo runner that trains and then runs the real-time simulation automatically.



## Abstract :
Intrusion detection system (IDS) has become an essential layer in all the latest ICT system due to an urge towards cyber safety in the day-to-day world. Reasons including uncertainty in ﬁnding the types of attacks and increased the complexity of advanced cyber attacks, IDS calls for the need of integration of Deep Neural Networks (DNNs). In this paper, DNNs have been utilized to predict the attacks on Network Intrusion Detection System (N-IDS). A DNN with 0.1 rate of learning is applied and is run for 1000 number of epochs and KDDCup-’99’ dataset has been used for training and benchmarking the network. For comparison purposes, the training is done on the same dataset with several other classical machine learning algorithms and DNN of layers ranging from 1 to 5. The results were compared and concluded that a DNN of 3 layers has superior performance over all the other classical machine learning algorithms. 

## Keywords : 
Intrusion detection, deep neural networks, machine learning, deep learning 

## Authors :
**[Rahul-Vigneswaran K](https://rahulvigneswaran.github.io)**<sup>∗</sup>, [Vinayakumar R](https://scholar.google.co.in/citations?user=oIYw0LQAAAAJ&hl=en&oi=ao)<sup>†</sup>, [Soman KP](https://scholar.google.co.in/citations?user=R_zpXOkAAAAJ&hl=en)<sup>†</sup> and [Prabaharan Poornachandran](https://scholar.google.com/citations?user=e233m6MAAAAJ&hl=en)<sup>‡</sup> 

**<sup>∗</sup>Department of Mechanical Engineering, Amrita Vishwa Vidyapeetham, India.** <br/> 
<sup>†</sup>Center for Computational Engineering and Networking (CEN), Amrita School of Engineering, Coimbatore.<br/> 
<sup>‡</sup>Center for Cyber Security Systems and Networks, Amrita School of Engineering, Amritapuri Amrita Vishwa Vidyapeetham, India.

## How to run the code?
### For **Classical Machine Learning**
* Run `all.py` [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/blob/master/all.py)
### For **Deep Neural Network (100 iterations)** 
* Run `dnn1.py` for 1-hidden layer network and run `dnn1acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn2.py` for 2-hidden layer network and run `dnn2acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn3.py` for 3-hidden layer network and run `dnn3acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn4.py` for 4-hidden layer network and run `dnn4acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)
* Run `dnn5.py` for 5-hidden layer network and run `dnn5acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn)

### For **Deep Neural Network (1000 iterations)** 
* Run `dnn1.py` for 1-hidden layer network and run `dnn1acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn2.py` for 2-hidden layer network and run `dnn2acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn3.py` for 3-hidden layer network and run `dnn3acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn4.py` for 4-hidden layer network and run `dnn4acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)
* Run `dnn5.py` for 5-hidden layer network and run `dnn5acc.py` for finding it's accuracy. [[Link]](https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/tree/master/dnn1000)



## Recommended Citation :
If you use this repository in your research, cite the the following papers :

  1. Rahul, V.K., Vinayakumar, R., Soman, K.P., & Poornachandran, P. (2018). Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security. 2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT), 1-6.
  2. Rahul-Vigneswaran, K., Poornachandran, P., & Soman, K.P. (2019). A Compendium on Network and Host based Intrusion Detection Systems. CoRR, abs/1904.03491.
  
  ### Bibtex Format :
```bib
@article{Rahul2018EvaluatingSA,
  title={Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security},
  author={Vigneswaran K Rahul and R. Vinayakumar and K. P. Soman and Prabaharan Poornachandran},
  journal={2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT)},
  year={2018},
  pages={1-6}
  }

@article{RahulVigneswaran2019ACO,
  title={A Compendium on Network and Host based Intrusion Detection Systems},
  author={K Rahul-Vigneswaran and Prabaharan Poornachandran and K. P. Soman},
  journal={CoRR},
  year={2019},
  volume={abs/1904.03491}
  }
```

## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase your are facing any difficulty with the code base or you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)]()

