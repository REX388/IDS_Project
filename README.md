# Intrusion Detection System (IDS) Project

## Overview
This project implements a Python-based Intrusion Detection System using machine learning algorithms. It simulates real-time intrusion detection on network traffic data, classifying samples as "Normal" or "Attack" using a trained Logistic Regression model and other classifiers.

## Features
- Uses KDD dataset for training and testing
- Multiple ML algorithms: Logistic Regression, Naive Bayes, KNN, Decision Tree, AdaBoost, Random Forest, SVM
- Real-time simulation script for streaming data classification
- Outputs model predictions and metrics to console and files

## File Structure
- `all.py`: Trains models, evaluates, and saves results
- `realtime_ids.py`: Simulates real-time IDS using trained model
- `kddtrain.csv`, `kddtest.csv`: Training and test data
- `classical/`: Stores trained models and output files
- `requirements.txt`: Python dependencies

## How to Run
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run model training and evaluation:
   ```powershell
   python all.py
   ```
3. Run real-time IDS simulation:
   ```powershell
   python realtime_ids.py
   ```

## Expected Output
- Console displays accuracy, precision, recall, and F1-score for each model
- Real-time alerts: "Normal" or "Attack" for each sample in simulation
- Output files in `classical/` directory

## Requirements
See `requirements.txt` for all required Python packages.

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

