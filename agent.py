import zmq
import random
import time
import rospy
from std_msgs.msg import String, Float32
import sensor_msgs
import ros_numpy
import torch
import torch.nn as nn,enisum    
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward

import argparse



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
EPISODE_STEPS = 100




class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, outputs):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.dqn_fc1 = nn.Linear(num_classes, 256)
        self.dqn_fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.mlp_head(x)
        x = F.relu(self.dqn_fc1(x))
        x = self.dqn_fc2(x)

        return x



class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)


        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent():
    def __init__(self):
        self.state = None
        self.done = None
        self.reward = 0
        context = zmq.Context()
        self.action_sender = context.socket(zmq.PUSH)
        self.action_sender.bind("tcp://127.0.0.1:5557")
        self.state_reciever = context.socket(zmq.PULL)
        self.state_reciever.connect("tcp://127.0.0.1:5558")
        self.slam_interact = rospy.Publisher('/ORB_SLAM/AGENT_SLAM', String, queue_size=10)
        rospy.Subscriber("/ORB_SLAM/Episode", String, self.update_episode)
        rospy.Subscriber("/ORB_SLAM/State", sensor_msgs.msg.Image, self.update_observation)
        rospy.Subscriber("/ORB_SLAM/Reward", Float32, self.update_reward)

        self.actions = ['forwards', 'backwards', 'turnLeft', 'turnRight', 'strafeRight', 'strafeLeft']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = len(self.actions)
        img = torch.ones([1, 32, 3, 224, 224]).cuda()
    
        self.policy_net = ViViT(224, 32, 100, 16,self.n_actions).cuda()
        self.target_net = ViViT(224, 32, 100, 16,self.n_actions).cuda()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.rcvd_state = False
        self.rcvd_reward = False
        self.rcvd_done = False


        self.steps_done = 0

    def select_action(self,state, inference=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold or inference:
            with torch.no_grad():
                outputs = self.policy_net(state)
                print("Estimated Q values:", self.actions[0], outputs[0][0].item(), self.actions[1], outputs[0][1].item(), self.actions[2], outputs[0][2].item(), self.actions[3], outputs[0][3].item(), self.actions[4], outputs[0][4].item(), self.actions[5], outputs[0][5].item())
                return outputs.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def initialize(self):


        print("Initializing the Map ..")

        t = 0

        while(self.done == "initializing"):
            self.rcvd_done = False
            self.rcvd_reward = False
            self.rcvd_state = False
            action = random.choice(self.actions[:2] + ["idle"])
            self.action_sender.send_string(action)
            while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state):
                pass
            t += 1
            if(t == 100):
                self.action_sender.send_string("reset")
                self.slam_interact.publish("reset")
                t = 0
                time.sleep(10)

        print("Finished Initializing the map")

    def restore(self):
        print("Loading pretrained model")
        state_dict = torch.load("agent.pt")
        self.policy_net.load_state_dict(state_dict["weights"])
        self.target_net.load_state_dict(state_dict["weights"])
        self.steps_done = state_dict["steps_done"]


    def train(self, resume):
        num_episodes = 200
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((80,80), interpolation=Image.CUBIC),
                            T.ToTensor()])

        if(resume):
            self.restore()
        
        for i_episode in range(num_episodes):
            # Initialize the environment and state

            self.initialize()

            print("Started Episode {}".format(i_episode))

            state = self.state


            avg_loss = 0
            total_reward = 0

            for t in count():
                # Select and perform an action
                state = resize(state).to(self.device).unsqueeze(0)
                action_idx = self.select_action(state)
                action = self.actions[action_idx.item()]

                self.rcvd_done = False
                self.rcvd_reward = False
                self.rcvd_state = False

                self.action_sender.send_string(action)

                while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state):
                    pass

                total_reward += self.reward
                reward = torch.tensor([self.reward], device=self.device)

                print("Reward : {:.2f}".format(reward.item()))
                
                # Observe new state
    
                if self.done != "initializing":
                    next_state = resize(self.state).to(self.device).unsqueeze(0)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action_idx, next_state, reward)

                # Move to the next state
                state = self.state

                # Perform one step of the optimization (on the target network)
                loss = self.optimize_model()
                if(loss):
                    avg_loss += loss
                if self.done == "initializing" or t == EPISODE_STEPS:
                    avg_loss = avg_loss / (t+1)
                    avg_reward = total_reward / (t+1)
                    self.writer = open("data.csv", "a+")
                    self.writer.write("{},{},{}\n".format(avg_loss, avg_reward, total_reward))
                    self.writer.close()

                    print("Episode {}, Avg_loss = {:.4f}, Avg reward = {:.2f}, total reward = {}".format(i_episode +1, avg_loss, avg_reward, total_reward))
                    self.action_sender.send_string("reset")
                    self.slam_interact.publish("reset")
                    time.sleep(5)
                    break

            torch.save({"weights": self.policy_net.state_dict(), "steps_done": self.steps_done}, "agent.pt")

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def inference(self):
        while True:
            resize = T.Compose([T.ToPILImage(),
                                T.Resize((80,80), interpolation=Image.CUBIC),
                                T.ToTensor()])
                # Initialize the environment and state

            self.initialize()

            state = self.state

            self.policy_net.load_state_dict(torch.load("agent.pt")["weights"])

            print("Loaded the agent successfully")

            for t in count():
                # Select and perform an action
                state = resize(state).to(self.device).unsqueeze(0)
                action_idx = self.select_action(state, inference=True)
                action = self.actions[action_idx.item()]

                self.rcvd_done = False
                self.rcvd_reward = False
                self.rcvd_state = False

                self.action_sender.send_string(action)

                while(not self.rcvd_done or not self.rcvd_reward or not self.rcvd_state):
                    pass

                reward = torch.tensor([self.reward], device=self.device)

                #print("Reward : {:.2f}".format(reward.item()))

                state = self.state

                if self.done == "initializing":
                    print("SLAM FAILURE HAPPENED")
                    break


    def update_episode(self,data):
        self.done = data.data
        self.rcvd_done = True

    def update_reward(self, data):
        self.reward = data.data
        self.rcvd_reward = True

    def update_observation(self, data):
        self.state = ros_numpy.numpify(data)
        self.rcvd_state = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=bool, help="Training the agent or running the pretrained one", default=False)
    parser.add_argument("--resume", type=bool, help="Training the agent from scratch or loading a pretrained model", default=False)
    args = parser.parse_args()
    rospy.init_node('agent')
    agent = Agent()
    if args.test:
        print("Interacting with the trained agent ...")
        agent.inference()
    else:
        print("Training the agent from scratch ..")
        agent.train(args.resume)

if __name__ == '__main__':
    main()