# Fake News Detection using Deep Learning

## Overview
This project focuses on detecting fake news using the Welfake dataset which contains a large collection of both real and fake news articles. The main goal was to apply and compare multiple deep learning architectures for binary text classification in order to identify which performs best at distinguishing between truthful and deceptive content.

Four models were implemented GRU BiGRU LSTM and a hybrid CNN+LSTM. Each model was trained and evaluated individually to measure accuracy precision recall f1-score and in some cases ROC-AUC. The GRU model achieved 98 percent accuracy with a ROC-AUC score of 0.9978 showing outstanding predictive power. The BiGRU model matched the 98 percent accuracy while maintaining balanced precision and recall. The CNN+LSTM model also achieved 98 percent accuracy successfully combining convolutional layers for feature extraction with LSTM layers for sequence learning. The LSTM model performed slightly lower at 97 percent accuracy but still demonstrated strong capability in handling sequential text data.

Across all experiments the models maintained high precision and recall values between 0.97 and 0.99 indicating reliable classification for both real and fake news. GRU and BiGRU stood out with a slight performance advantage suggesting that gated recurrent architectures are particularly well suited for this task. These results reinforce the potential of deep learning as an effective tool for combating misinformation in digital media.

## Dataset
Name: Welfake Dataset  
Description: A large labeled dataset containing thousands of real and fake news articles intended for binary classification research.  
Source: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification  

## Models Implemented
1. GRU – Gated Recurrent Unit a lightweight recurrent neural network for efficient sequence modeling  
2. BiGRU – Bidirectional GRU which processes input in both forward and backward directions  
3. LSTM – Long Short-Term Memory network specialized for learning long-range dependencies  
4. CNN+LSTM – A hybrid network combining convolutional layers for feature extraction with LSTM layers for temporal context learning  

## Results Summary
Model     | Accuracy | Precision | Recall | F1-score | ROC-AUC  
GRU       | 98%      | ~0.98     | ~0.98  | ~0.98    | 0.9978  
BiGRU     | 98%      | ~0.98     | ~0.98  | ~0.98    | -  
CNN+LSTM  | 98%      | ~0.98     | ~0.98  | ~0.98    | -  
LSTM      | 97%      | ~0.97     | ~0.97  | ~0.97    | -  

## Confusion Matrices
GRU: [[6762 244],[75 7346]]  
BiGRU: [[6892 114],[157 7264]]  
CNN+LSTM: [[6866 140],[201 7220]]  
LSTM: [[6670 336],[83 7338]]  
