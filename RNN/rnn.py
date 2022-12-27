import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

input_size = 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
sequence_length = 28
num_epochs = 2
num_layers = 2
hidden_size = 256

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size	
		self.num_layers = num_layers
		self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
		# Nxt_time 
		self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		out,_ = self.rnn(x, h0)
		out = out.reshape(out.shape[0], -1) 
		out = self.fc(out) # take information from all the hidden states


if __name__ == "__main__":
	print("Being run as the main module")
	x = torch.randn(28, 28)
