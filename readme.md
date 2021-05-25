
# Sample pytorch code for

BEAN: Interpretable and Efficient Learning with Biologically-Enhanced Artificial Neuronal Assembly Regularization

# Desciption

This codebase provides the basic runing environment for LeNet-5 + BEAN on MNIST dataset

main.py -> Model training & testing on the full MNIST dataset

main_fewshot.py -> Model training & testing with few-shot learning setup on the MNIST dataset

#  installation

Python pakage requirement:
- python==3.7.9
- pytorch==1.5.0 
- torchversion==0.6.0

For more information about installation, please find here:
https://pytorch.org/get-started/previous-versions/#v150

# How to run (sample script for code runing)

Example 1. Train a LeNet-5 + BEAN-1 model on the full MNIST dataset

$ python main.py

Example 1. Train a LeNet-5 + BEAN-1 model on a 10-shot learning setup (only use 10 training samples per class, i.e. total of 100 samples of out 50000) on the MNIST dataset

$ python main_fewshot.py --num_epochs 100 --batch_size 1 --learning_rate 0.0005 --n_shot 10 --seed 1 --BEAN 2 --alpha 100

More tips to reproduce results:

- Modify [seed] to yeild different random data sample sleection from the full dataset.

- Modify [BEAN] to 1 or 2 to test the two varations of BEAN regualrization
- Set [alpha] to be 0 to see the base model performance without BEAN regularization

- Modify [n_shot] to test model performance on different n-shot learning
- For small n such as the extreme 1-shot learning, it might takes more [num_epochs] for the model to converge, consider increase the [num_epochs] to 200 and monitoring the model performance on validation set. 

If you find this code useful in your research, please consider cite our paper:

> @article{gao2021bean,
  title={BEAN: Interpretable and Efficient Learning with Biologically-Enhanced Artificial Neuronal Assembly Regularization},
  author={Gao, Yuyang and Ascoli, Giorgio and Zhao, Liang},
  journal={Frontiers in Neurorobotics},
  volume={15},
  pages={68},
  year={2021},
  publisher={Frontiers}
  }
