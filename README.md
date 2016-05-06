# SPiCe Project

###### Zhengli Zhao, Karthik Prasad, Abhisaar Sharma

This code is for Sequence PredictIction ChallengE (SPiCe) which aims to guess the next element in a sequence through learning a model that allows the ranking of potential next symbols for a given prefix. The competition uses both real-world and synthetic data. Given the first prefix of these test sets we have to submit a ranking of the 5 most probable next symbols to be fed with the following prefix of the test set.

The primary aim of this project will be understand the different techniques that can be applied to this graphical model and work towards enhancing the accuracy of the prediction. 

We have tested the following techniques:
- Spectral Baseline
- 3 gram
- Gibbs sampling

### Executing
The program is written in python 
```sh
python codeBlue.py
```

