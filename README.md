# LSTM
A simple LSTM with one hidden state, based off the work that was done for RNN.py and works in a similar fashion. 

## How It Works
The network works in two steps. The first step involves initializing the network and the second step involves training the network. 
<p>For step 1 we simply make the following declarion:
rnn = RNN(data, hidden_size, eps=0.0001)

<p> Step 2 involes training the network and is performed by calling `train(self, seq_length, epochs, eta, decay_rate=0.9, learning_decay=0.0, randomize=False, print_final=True)` and selecting values for the parameters: 

<p><b>seq_length</b>: Integer value for the desired length of the subsubsequence<br> 
<b>epochs</b>: Integer value for the number of iteration to train over.<br>
<b>eta</b>: Learing rate for gradient descent.<br>
<b>decay_rate</b>: Decay parameter for the moving average. The value must lie between [0, 1) where smaller values indicate shorter memory. The default value is set to `0.9`<br>
<b>learning_decay</b>: Annealing parameter for the exponetial decay. The smaller values indicate milder annealing of the learning rate. The default value is set to `0.0`<br>
<b>randomize</b>: If set to `True` then the subsequences will be shuffle before beign processed further. The default value is set to `False`. <br>
<b>print_final</b>: Prints the final output at the end of evey epoch. The default value is set to `True`. <br>
