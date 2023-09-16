import os
import gym
import torch
import argparse
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-12

timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
writer = SummaryWriter(Path("./runs", f'{timestamp}'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
class Policy(nn.Module):
    def __init__(self, hid_dim=16):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hid_dim = hid_dim
        self.double()

        self.input_layer = nn.Sequential(nn.Linear(self.observation_dim, self.hid_dim), nn.ReLU())
        self.p_layer1 = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU())
        self.p_layer2 = nn.Linear(self.hid_dim, self.action_dim)
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = self.input_layer(state)
        out = self.p_layer1(x)
        out = self.p_layer2(out)
        action_prob = F.softmax(out, dim=-1)

        return action_prob


    def select_action(self, state):
        """
            Select the action given the current state.
        """
        state = torch.tensor(state).float().to(device) 
        action_prob = self.forward(state) 
        dist = Categorical(action_prob) # convert to a distribution
        action = dist.sample() # choose action from the distribution
        
        self.saved_actions.append(dist.log_prob(action)) # save to action buffer

        return action.item()
    
    def saved_rewards(self, reward):
        self.rewards.append(reward)


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss based on the collected rewards using REINFORCE.
        """
        saved_actions = self.saved_actions # list of actions
        rewards = self.rewards # list of rewards
        policy_losses = [] 
        returns = []

        for t in range(len(rewards)-1, -1, -1):
            disc_returns = (returns[0] if len(returns)> 0 else 0)
            returns.insert(0, gamma * disc_returns + rewards[t]) # insert in the beginning of the list 
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS) # to stabilize training

        for step in range(len(saved_actions)):
            log_prob = saved_actions[step]
            G = returns[step]
            policy_losses.append(G * log_prob)

        policy_loss = torch.stack(policy_losses, dim=0).sum()
        loss = -policy_loss # make the loss negative to do gradient descent
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(args):
    model = Policy(args.hid_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    ewma_reward = 0  # EWMA reward for tracking the learning progress
    
    for episode in range(args.episodes):
        state = env.reset() # reset environment and episode reward
        ep_reward = 0
        t = 0
        steps = 9999 # set to avoid infinite loop

        for t in range(steps):
            action = model.select_action(state=state)
            state, reward, done, info = env.step(action)
            model.saved_rewards(reward)
            ep_reward += reward
            if done: break
        
        loss = model.calculate_loss(args.gamma)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.clear_memory()
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        if (episode+1) % 10 == 0:
            print(f"Episode {episode}\tlength: {t+1}\treward: {ep_reward}\t ewma reward: {ewma_reward}")

        writer.add_scalar("Train/episode length", t, episode+1)
        writer.add_scalar("Train/loss", -loss, episode+1)
        writer.add_scalars("Train/reward", {"episode reward": ep_reward, "ewma reward": ewma_reward}, episode)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], episode+1)

        if ewma_reward > env.spec.reward_threshold or episode == args.episodes-1:
            if not os.path.isdir("./models"):
                os.mkdir("./models")
            torch.save(model.state_dict(), f"./models/{args.env}.pth")
            break


def test(args, model_name): 
    model = Policy(args.hid_dim).to(device)   
    model.load_state_dict(torch.load(f"./models/{model_name}"))

    max_episode_len = 10000
    
    state = env.reset()
    running_reward = 0
    for t in range(max_episode_len+1):
        action = model.select_action(state)
        state, reward, done, info = env.step(action)
        running_reward += reward
        if done:
            break
    print(f"Testing: Reward: {running_reward}")
    env.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("REINFORCE algorithm")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Name of the environment")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--hid_dim", type=int, default=16, help="Hidden dimension of the policy network")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes for training")
    args = parser.parse_args()

    random_seed = args.seed 
    env = gym.make(args.env)
    env.seed(random_seed)
    torch.manual_seed(random_seed)  

    train(args)
    test(args, f"{args.env}.pth")


    