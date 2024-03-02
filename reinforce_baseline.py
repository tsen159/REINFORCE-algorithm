import os
import gym
import torch
import argparse
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-12

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
writer = SummaryWriter(Path("./runs", f'{timestamp}'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
    """
    def __init__(self, hid_dim=16):
        super(Policy, self).__init__()
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hid_dim = args.hid_dim
        self.double()
        
        self.input_layer = nn.Sequential(nn.Linear(self.observation_dim, self.hid_dim), nn.ReLU())
        self.p_layer1 = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU())
        self.p_layer2 = nn.Linear(self.hid_dim, self.action_dim)

        self.v_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU()) 
             for _ in range(2)])
        self.v_output_layer = nn.Linear(self.hid_dim, 1)
           
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = self.input_layer(state)
        out = self.p_layer1(x)
        out = self.p_layer2(out)
        action_prob = F.softmax(out, dim=-1)

        for layer in self.v_layers:
            value = layer(x)
        state_value = self.v_output_layer(value)

        return action_prob, state_value


    def select_action(self, state):

        state = torch.tensor(state).float().to(device) 
        action_prob, state_value = self.forward(state) 
        dist = Categorical(action_prob) # convert to a distribution
        action = dist.sample() # choose action from the distribution
        
        self.saved_actions.append(SavedAction(dist.log_prob(action), state_value)) # save to action buffer

        return action.item()
    
    def saved_rewards(self, reward):
        self.rewards.append(reward)


    def calculate_loss(self, gamma=0.999):
        saved_actions = self.saved_actions # list of actions
        rewards = self.rewards # list of rewards
        policy_losses = [] 
        state_value_list = [] 
        returns = []
        adv_list = []

        for t in range(len(rewards)-1, -1, -1): # calculate disounted returns in each time step
            disc_returns = (returns[0] if len(returns)> 0 else 0)
            G_t = gamma * disc_returns + rewards[t]
            returns.insert(0, G_t) # insert in the beginning of the list
            state_value = saved_actions[t][1]
            state_value_list.append(state_value)
            adv_list.insert(0, G_t - state_value)

        adv_list = torch.tensor(adv_list)
        adv_list = (adv_list - adv_list.mean()) / (adv_list.std() + EPS) # for stability

        for step in range(len(saved_actions)):
            log_prob = saved_actions[step][0]
            adv = adv_list[step]
            policy_losses.append(adv * log_prob)

        value_loss = F.mse_loss(torch.tensor(state_value_list), torch.tensor(returns))
        policy_loss = torch.stack(policy_losses, dim=0).sum()
        loss = -policy_loss + value_loss 
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(args):
    model = Policy(hid_dim=args.hid_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)
    
    ewma_reward = 0  # EWMA reward for tracking the learning progress

    for episode in range(args.episodes):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        
        steps = 9999
        for t in range(steps):
            action = model.select_action(state=state)
            state, reward, done, _ = env.step(action)
            model.saved_rewards(reward)
            ep_reward += reward
            if done: break
            
        loss = model.calculate_loss(gamma=args.gamma)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.clear_memory()
        
        # update EWMA reward and log the results 
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}\tlength: {t+1}\treward: {ep_reward}\t ewma reward: {ewma_reward}")

        writer.add_scalar("Train/episode length", t, episode)
        writer.add_scalar("Train/loss", -loss, episode)
        writer.add_scalars("Train/reward", {"episode reward": ep_reward, "ewma reward": ewma_reward}, episode)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], episode)


        if ewma_reward > env.spec.reward_threshold or episode == args.episodes-1:
            if not os.path.isdir("./models"):
                os.mkdir("./models")
            torch.save(model.state_dict(), f"./models/{args.env}_baseline.pth")
            break


def test(args, model_name): 
    model = Policy(hid_dim=args.hid_dim).to(device)   
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
    parser = argparse.ArgumentParser("REINFORCE algorithm using baseline")
    parser.add_argument("--env", type=str, default="LunarLander-v2", help="Name of the environment")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=200, help="Step size for lr scheduler")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of episodes for training")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--hid_dim", type=int, default=16, help="Hidden dimension of the policy network")
    
    args = parser.parse_args()

    random_seed = args.seed 
    lr = args.lr
    hid_dim = args.hid_dim
    env = gym.make(args.env)
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  

    train(args)
    test(args, f'{args.env}_baseline.pth')