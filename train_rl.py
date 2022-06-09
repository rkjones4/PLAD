import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import time
import utils
from utils import device
from torch.autograd.variable import Variable

# policy gradient implementation based off of the one presented in CSGNet: https://github.com/Hippogriff/CSGNet

# Class to keep track of recent rewards 
class Reinforce:
    def __init__(self):

        self.alpha_baseline = 0.7
        self.rolling_baseline = Variable(torch.zeros(1)).to(device)
        self.max_reward = Variable(torch.zeros(1)).to(device)

    # calculate policy gradient loss
    def pg_loss_var(self, R, probs, samples, len_programs):
        
        batch_size = R.shape[0]
        R = Variable(R).cuda().view(batch_size, 1)

        samples = [s.data.cpu().numpy() for s in samples]
        
        self.rolling_baseline = self.alpha_baseline * self.rolling_baseline + \
            (1 - self.alpha_baseline) * torch.mean(R)
            
        baseline = self.rolling_baseline.view(1, 1).repeat(batch_size, 1)
        baseline = baseline.detach()
        advantage = R - baseline
        
        temp = []
        for i in range(batch_size):
            neg_log_prob = Variable(torch.zeros(1)).cuda()
            # Only summing the probs before stop symbol
            for j in range(len_programs[i]):
                neg_log_prob = neg_log_prob + probs[j][i, samples[j][i]]

            temp.append(neg_log_prob / (len_programs[i] * 1.))

        loss = -torch.cat(temp).view(batch_size, 1)
        loss = loss.mul(advantage)
        loss = torch.mean(loss)
        
        return loss
                
def train_rec(net, cad_data, args, domain):
    epochs = args.epochs
    path = args.infer_path

    train_gen = cad_data.train_rl_iter()
    val_gen = cad_data.val_eval_iter
    
    optimizer = optim.SGD(
        net.parameters(),
        momentum=0.9,
        lr=args.lr,
        nesterov=False
    )
    
    torch.save(net.state_dict(), f"{path}/best_dict.pt")
    
    num_traj = args.num_traj
    
    for epoch in range(epochs):
        
        start = time.time()

        train_loss = 0
        total_reward = 0

        net.train()
        net.epsilon = 1.0

        train_loss = []
        rewards = []
        
        for batch_idx in range(
            args.train_size //
            (args.batch_size * args.num_traj)
        ):
            optimizer.zero_grad()

            # only make gradient update after a certrain number of batches (trajectories)
            
            for _ in range(args.num_traj):
                
                voxels = next(train_gen)                                
                
                outputs, samples = net.rl_fwd(voxels)

                R, prog_lens = net.generate_rewards(
                    samples,
                    voxels
                )
                
                loss = domain.reinforce.pg_loss_var(R, outputs, samples, prog_lens) / (num_traj * 1.)

                loss.backward()
                
                rewards.append(R.mean().item())
                train_loss.append(loss.item() * num_traj)
                
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
            
                                
        mean_train_loss = round(torch.tensor(train_loss).mean().item(), 4)
        mean_reward = round(torch.tensor(rewards).mean().item(), 4)

        net.eval()
        net.epsilon = 0
        metric = []
        
        with torch.no_grad():
            for vinput in val_gen():
                _, _, pred_metric = net.eval_infer_progs(vinput, args.es_beams)
                metric += pred_metric

        METRIC = torch.tensor(metric).mean().item()
        
        net.eval()
        net.epsilon = 0

        end = time.time()
        utils.log_print(f"Epoch {epoch}/{epochs} =>  loss: {mean_train_loss}, reward: {mean_reward}, val metric: {METRIC} | {end-start}", args)        

    torch.save(net.state_dict(), f"{path}/best_dict.pt")
    
    return epochs+1
