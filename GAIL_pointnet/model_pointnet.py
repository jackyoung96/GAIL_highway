import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc0 = nn.Linear(num_inputs, args.hidden_size_1) # 7개 feature, 100개의 hidden size라고 가정 -> 이거는 share할 수 있도록
        self.mp = nn.MaxPool1d(num_inputs)
        self.fc1 = nn.Linear(args.hidden_size_1, args.hidden_size_2) # 100*차량 개수 -> 100이 되는 것.
        self.fc2 = nn.Linear(args.hidden_size_2, num_outputs)
        # self.fc3 = nn.Linear(args.hidden_size, num_outputs)
        
        self.fc2.weight.data.mul_(0.1)
        self.fc2.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        x = torch.transpose(x,-2,-1)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim = 0)
        x = self.mp(x).squeeze() # max pooling
        x = torch.tanh(self.fc1(x))
        mu = F.softmax(self.fc2(x), dim=-1)
        # logstd = torch.zeros_like(mu) # 0짜리를 왜만드는 건지 모르겠다.
        # std = torch.exp(logstd) # 1짜리는 또 왜 만드는 걸까 -> continous action이라서 그렇다.
        return mu #, std


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc0 = nn.Linear(num_inputs, args.hidden_size_1) # 7개 feature, 100개의 hidden size라고 가정 -> 이거는 share할 수 있도록
        self.mp = nn.MaxPool1d(num_inputs)
        self.fc1 = nn.Linear(args.hidden_size_1, args.hidden_size_2) # 100*차량 개수 -> 100이 되는 것.
        self.fc2 = nn.Linear(args.hidden_size_2, 1)
        
        self.fc2.weight.data.mul_(0.1)
        self.fc2.bias.data.mul_(0.0)

    def forward(self, x):
        # x = Variable(x, requires_grad=True)
        x = torch.tanh(self.fc0(x))
        x = torch.transpose(x,-2,-1)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim = 0)
        x = self.mp(x).squeeze() # max pooling
        x = torch.tanh(self.fc1(x))
        v = self.fc2(x)  # Q value는 negative도 가능
        return v


class Discriminator(nn.Module):
    def __init__(self, num_state_inputs, num_action_inputs, args):
        super(Discriminator, self).__init__()
        self.fc0 = nn.Linear(num_state_inputs, args.hidden_size_1) # 7개 feature, 100개의 hidden size라고 가정 -> 이거는 share할 수 있도록
        self.mp = nn.MaxPool1d(num_state_inputs)
        self.fc1 = nn.Linear(args.hidden_size_1 + num_action_inputs, args.hidden_size_2)
        self.fc2 = nn.Linear(args.hidden_size_2, 1)
        
        self.fc2.weight.data.mul_(0.1)
        self.fc2.bias.data.mul_(0.0)

    def forward(self, x, a):
        
        x = torch.tanh(self.fc0(x))
        x = torch.transpose(x,-2,-1)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim = 0)
        x = self.mp(x).squeeze() # max pooling
        x = torch.cat((x,a), dim=-1)
        x = torch.tanh(self.fc1(x))
        # print("discriminator: ",x)
        prob = torch.sigmoid(self.fc2(x))
        # print("discriminator_p: ",prob)
        return prob

# actor = Actor()
# s = torch.tensor([])

# parser = argparse.ArgumentParser(description='PyTorch GAIL')

# parser.add_argument('--hidden_size_1', type=int, default=100, 
# 					help='hidden unit size of actor, critic and discrim networks (default: 100)')
# parser.add_argument('--hidden_size_2', type=int, default=100, 
# 					help='hidden unit size of actor, critic and discrim networks (default: 100)')                    

# args = parser.parse_args()

# actor = Critic(3, args)
# s = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]).float()
# output =actor(s)
# print(output)

# Dis = Discriminator(3, 5, args)
# sta = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]).float()
# act = torch.tensor([0.1,0.2,0.3,0.4,0.0])
# print(Dis(sta,act))