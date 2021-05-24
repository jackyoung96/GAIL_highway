import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)
        
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = F.softmax(self.fc3(x), dim=-1)
        # logstd = torch.zeros_like(mu) # 0짜리를 왜만드는 건지 모르겠다.
        # std = torch.exp(logstd) # 1짜리는 또 왜 만드는 걸까 -> continous action이라서 그렇다.
        return mu #, std


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        # x = Variable(x, requires_grad=True)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # print("discriminator: ",x)
        prob = torch.sigmoid(self.fc3(x))
        # print("discriminator_p: ",prob)
        return prob

# actor = Actor()
# s = torch.tensor([])