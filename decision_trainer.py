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
        s0, x0_static, a, r, s1, x1_static , done = zip(*random.sample(self.buffer, batch_size))

        # arr1 = np.array(s0)
        # return np.concatenate(s0), a, r, np.concatenate(s1), done   #(32, 6, 96, 96)
        return s0, x0_static, a, r, s1, x1_static , done

class decision_driving_Agent:

    def __init__(self,inputs_shape,num_actions,is_training,ROI_length,extra_num):
        self.extra_num = extra_num
        self.inputs_shape = inputs_shape
        self.num_actions = num_actions
        self.is_training = is_training
        self.batch_size = 1000
        self.selection_method = None
        self.gamma = 0.999
        self.buffer = ReplayBuffer()
        self.model  = DQN(self.inputs_shape,20,3,80,self.num_actions,self.batch_size,self.extra_num).cuda()
        # self.model2 =
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.decaying = 0.999
        self.learning_rate =0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = 9999999999
        self.ROI_length = ROI_length #(meters)

    def search_extra_in_ROI(self,extra_lists,player,extra_dl_list):
        new_extra_lists = []
        new_extra_dl_lists = []
        ROI_distance = 1000 #(m)
        for n,extra in enumerate(extra_lists):
            distance_between_vehicles = ((extra.get_location().x-player.get_location().x)**2+(extra.get_location().y-player.get_location().y)**2+(extra.get_location().z-player.get_location().z)**2)**0.5
            if distance_between_vehicles < ROI_distance:
                new_extra_lists.append(extra)
                new_extra_dl_lists.append(extra_dl_list[n])
        out = [new_extra_lists, new_extra_dl_lists]
        return out

    def act(self,state,x_static):

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decaying
        else:
            self.epsilon = self.epsilon_min

        # x_static = state[-1]
        # state = state[:-1]

        if random.random() > self.epsilon or not self.is_training:
            self.selection_method = 'max'
            state = torch.tensor(state, dtype = torch.float).cuda() #uint8 ->float
            x_static = torch.tensor(x_static, dtype = torch.float).cuda()
            self.q_value = self.model(state,x_static)
            action =  int(self.q_value.argmax().item())-1

            print("Q: ",self.q_value, "ACTION :",action)

            # print(self.q_value)



        else:
            self.selection_method = 'random'
            action = random.randrange(self.num_actions)-1
            print("random action :", action)
        return action

    def learning(self): # sample에서 뽑고 backpropagation 하는 과정
        """
        DQN learning
        """
        s0, x0_static, a, r, s1, x1_static, done = self.buffer.sample(self.batch_size)
        # print(s0)

        s0 = torch.tensor(s0, dtype=torch.float)
        x0_static = torch.tensor(x0_static, dtype= torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        x1_static = torch.tensor(x1_static, dtype = torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        s0 = s0.cuda()
        x0_static = x0_static.cuda()
        s1 = s1.cuda()
        x1_static = x1_static.cuda()
        a = a.cuda()
        r = r.cuda()
        done = done.cuda()

        ##  forward  ##
        # print("start forward")
        q_values = self.model(s0,x0_static).cuda()
        next_q_values = self.model(s1,x1_static).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)+1).squeeze(1)
        # print(next_q_values)
        next_q_value = next_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_values = r + self.gamma * next_q_value * (1 - done)
        # print("expected_q_values :" , expected_q_values[0], "q_value :", q_value[0])
        # print("finished forward")

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

    def D3QN_learning(self):
        """
                DQN learning
                """
        s0, x0_static, a, r, s1, x1_static, done = self.buffer.sample(self.batch_size)
        # print(s0)

        s0 = torch.tensor(s0, dtype=torch.float)
        x0_static = torch.tensor(x0_static, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        x1_static = torch.tensor(x1_static, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        s0 = s0.cuda()
        x0_static = x0_static.cuda()
        s1 = s1.cuda()
        x1_static = x1_static.cuda()
        a = a.cuda()
        r = r.cuda()
        done = done.cuda()

        ##  forward  ##
        # print("start forward")
        q_values = self.model(s0, x0_static).cuda()
        next_q_values = self.model(s1, x1_static).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1) + 1).squeeze(1)
        # print(next_q_values)
        next_q_value = next_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_values = r + self.gamma * next_q_value * (1 - done)
        # print("expected_q_values :" , expected_q_values[0], "q_value :", q_value[0])
        # print("finished forward")

        # 0 : 좌회전 , 1 : 직진 : 2 : 우회전 시 Q value
        self.loss = (q_value - expected_q_values.detach()).pow(2).mean()

        # print(" loss : ", self.loss, "q_value :", q_value.mean())
        # next_q_state_values = self.target_model(s1).cuda()

        # off-policy

        ##  backward  ##
        self.loss.backward()

        ##  update weights  ##
        self.optimizer.step()

        # zero the gradients after updating
        self.optimizer.zero_grad()

    def cuda(self):
        self.model.cuda()
        # self.target_model.cuda()

# class phi_network(nn.Module):
#     def __init__(self,input_size, hidden_size,feature_size):
#         super(phi_network,self).__init__()
#         self.input_size = input_size
#         self.l1 = nn.Linear(input_size,hidden_size)
#         self.l2 = nn.Linear(hidden_size,hidden_size)
#         self.l3 = nn.Linear(hidden_size,feature_size)
#         self.relue = nn.ReLU()
#     def forward(self,x):
#         x.view(-1)
class DQN(nn.Module):
    def __init__(self,input_size,feature_size,x_static_size, hidden_size, output_size,batch_size,extra_num):
        super(DQN,self).__init__()
        self.extra_num = extra_num
        self.input_size = input_size
        self.static_size = x_static_size
        self.feature_size = feature_size+x_static_size
        self.batch_size = batch_size
        self.phi_network = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,feature_size)
        )
        self.rho_network = nn.Sequential(
            nn.Linear(feature_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,feature_size)
        )
        self.l1 = nn.Linear(self.feature_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self,x,x_static):
        # print(x.size(0))

        x=x.view(-1,self.extra_num, self.input_size)
        x_static= x_static.view(-1, self.static_size)
        # feature_points=torch.zeros(self.feature_size-self.static_size).cuda()
        # for index in x:
        #     feature_points+=self.phi_network(index)
        feature_points = self.phi_network(x)
        feature_points_sum = torch.sum(feature_points,1).squeeze(1)
        out = self.rho_network(feature_points_sum)
        # print("before concat :", out.shape)
        out = torch.cat((out,x_static),1)
        # print("after concat :", out.shape)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # out = self.softmax(out)
        return out

class D3QN(nn.Module):
    def __init__(self, input_size, feature_size, x_static_size, hidden_size, output_size, batch_size, extra_num):
        super(DQN, self).__init__()
        self.extra_num = extra_num
        self.input_size = input_size
        self.static_size = x_static_size
        self.feature_size = feature_size + x_static_size
        self.batch_size = batch_size
        self.phi_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )
        self.rho_network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )
        self.l1 = nn.Linear(self.feature_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_adv = nn.Linear(hidden_size, output_size)
        self.fc3_v = nn.Linear(hidden_size,1)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, x_static):
        # print(x.size(0))

        x = x.view(-1, self.extra_num, self.input_size)
        x_static = x_static.view(-1, self.static_size)
        # feature_points=torch.zeros(self.feature_size-self.static_size).cuda()
        # for index in x:
        #     feature_points+=self.phi_network(index)
        feature_points = self.phi_network(x)
        feature_points_sum = torch.sum(feature_points, 1).squeeze(1)
        out = self.rho_network(feature_points_sum)
        # print("before concat :", out.shape)
        out = torch.cat((out, x_static), 1)
        # print("after concat :", out.shape)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        adv = self.l3(out)
        val = self.fc3_v(out).expand(-1,adv.size(1))
        out = val + adv + adv.mean(1,keepdim=True).expand(-1,adv.size(1))
        # out = self.softmax(out)
        return out


