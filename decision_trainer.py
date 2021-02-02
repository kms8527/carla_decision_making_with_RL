import sys

tmp = ['/home/zz/carmaker_ros/ros/ros1_ws/devel/lib/python2.7/dist-packages',
       '/opt/ros/ros1/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages',
       '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla-0.9.6-py3.5-linux-x86_64.egg',
       '/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES']
for index in range(0, len(tmp)):
    if tmp[index] in sys.path:
        sys.path.remove(tmp[index])

if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg_FILES')

if not ('/home/zz/Downloads/CARLA_0.9.6/HDMaps' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/HDMaps')
if not ('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla' in sys.path):
    sys.path.append('/home/zz/Downloads/CARLA_0.9.6/PythonAPI/carla')

import time
import carla
import collections
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import numpy as np
import random



class ReplayBuffer:
    def __init__(self):
        self.buffer_limit = 10000
        self.buffer = collections.deque(maxlen = self.buffer_limit)

    def append(self,sample): #sample은 list
        self.buffer.append(sample)

    def size(self):
        return self.buffer

    def sample(self,batch_size):
        # s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))

        # arr1 = np.array(s0)
        # return np.concatenate(s0), a, r, np.concatenate(s1), done   #(32, 6, 96, 96)
        return s0, a, r, s1, done   #(32, 6, 96, 96)

class decision_driving_Agent:

    def __init__(self,inputs_shape,num_actions,is_training):

        self.inputs_shape = inputs_shape
        self.num_actions = num_actions
        self.is_training = is_training
        self.batch_size = 100
        self.selection_method = None
        self.gamma = 0.9
        self.buffer = ReplayBuffer()
        self.model  = DQN(self.inputs_shape,100,self.num_actions).cuda()
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.decaying = 0.999
        self.learning_rate =0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = 9999999999

    def act(self,state):

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decaying
        else:
            self.epsilon = self.epsilon_min

        if random.random() > self.epsilon or not self.is_training:
            self.selection_method = 'max'
            state = torch.tensor(state, dtype = torch.float).cuda() #uint8 ->float
            self.q_value = self.model(state)
            # print(self.q_value)
            action =  int(self.q_value.max().item())-1 #이게 argmax Q가 맞다고?
            # print(action)


        else:
            self.selection_method = 'random'
            action = random.randrange(self.num_actions)-1
        return action

    def learning(self): # sample에서 뽑고 backpropagation 하는 과정
        s0, a, r, s1, done = self.buffer.sample(self.batch_size)
        # print(s0)
        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        s0 = s0.cuda()
        s1 = s1.cuda()
        a = a.cuda()
        r = r.cuda()
        done = done.cuda()

        ##  forward  ##

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)+1).squeeze(1)
        next_q_value = next_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_values = r + self.gamma * next_q_value * (1 - done)

        # 0 : 좌회전 , 1 : 직진 : 2 : 우회전 시 Q value
        self.loss = (q_value - expected_q_values.detach()).pow(2).mean()

        # print(" loss : ", self.loss, "q_value :", q_value.mean())
        # next_q_state_values = self.target_model(s1).cuda()

        #off-policy


        ##  backward  ##
        self.loss.backward()

        ##  update weights  ##
        self.optimizer.step()

        # zero the gradients after updating
        self.optimizer.zero_grad()



    def cuda(self):
        self.model.cuda()
        # self.target_model.cuda()


class DQN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(DQN,self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,output_size)
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):
        # print(x.size(0))

        x.view(-1, self.input_size)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # out = self.softmax(out)
        return out




