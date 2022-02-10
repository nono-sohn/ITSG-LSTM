# ITSG-LSTM
This is a implementation of ITSG-LSTM: Inter-Country Trade Similarity Graph-Based Long Short-Term Memory for Port Throughput Prediction.

# Requirement
Python 3.7.9 

numpy >= 1.19.2

pandas >= 1.2.1

gensim >= 3.8.3

dgl >= 0.5.3

networkx >= 2.5

regex >= 2022.1.18

scikit-learn >= 0.24.1

torch >= 1.7.1

# Implementation
``` python main.py ```

# Arguments
--cargo : cargo type ex) LBC, DBC
--target : target port throughput ex) total(TOTAL THROUGHPUT), crude (CRUDE OIL)

--input_dim  
--gcn_dim
--lstm_dim
--output_dim

--epochs
--learning_rate

--cut_off : threshold for edge connection
--seq_length    

--window : hyperparameter of Word2vec model
--size : hyperparameter of Word2vec model

--split_ratio : train-test ratio
...

