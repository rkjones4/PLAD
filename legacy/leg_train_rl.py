import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from leg_src.Models.models import ImitateJoint, ParseModelOutput
from leg_src.Models.loss import losses_joint
from leg_src.utils.train_utils import prepare_input_op, chamfer, beams_parser
from leg_src.utils.generators.shapenet_generater import Generator
from leg_src.utils.reinforce import Reinforce
import time
import utils
from torch.autograd.variable import Variable
from utils import device

"""
Trains CSGNet to convergence on samples from generator network
"""

def train(csgnet, args):
    epochs = args.epochs
    path = args.infer_path
    max_len = args.max_len
    inference_train_size = args.train_size
    inference_test_size = args.test_size

    with open("legacy/terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]
    
    generator = Generator()
    reinforce = Reinforce(unique_draws=unique_draw)
    
    train_gen = generator.train_gen(
        batch_size=args.batch_size,
        path=args.data_path,
    )
    
    val_gen = generator.val_gen(
        batch_size=args.batch_size,
        path=args.data_path,
    )

    optimizer = optim.SGD(
        [para for para in csgnet.parameters() if para.requires_grad],
        weight_decay=args.weight_decay,
        momentum=0.9,
        lr=args.lr,
        nesterov=False
    )
    
    torch.save(csgnet.state_dict(), f"{path}/best_dict.pt")
    
    num_traj = args.num_traj
            
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        total_reward = 0
        csgnet.train()
        csgnet.epsilon = 1.0
        for batch_idx in range(inference_train_size //
                               (args.batch_size * args.num_traj)):
            optimizer.zero_grad()

            loss_sum = Variable(torch.zeros(1)).cuda().data
            Rs = np.zeros((args.batch_size, 1))
            
            for _ in range(args.num_traj):
                labels = np.zeros((args.batch_size, max_len), dtype=np.int32)
                data_ = next(train_gen)                                
                one_hot_labels = prepare_input_op(labels, len(unique_draw))                
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = Variable(torch.from_numpy(data_)).cuda()
                
                outputs, samples = csgnet.rl_fwd([data, one_hot_labels, max_len])

                R = reinforce.generate_rewards(
                    samples,
                    data_,
                    time_steps=max_len,
                    stack_size=max_len // 2 + 1,
                    reward='chamfer',
                    power=20
                )

                R = R[0]
                loss = reinforce.pg_loss_var(
                    R, samples, outputs) / num_traj
                loss.backward()
                Rs = Rs + R
                loss_sum += loss.data
                
            Rs = Rs / (num_traj)
            torch.nn.utils.clip_grad_norm_(csgnet.parameters(), 10)
            optimizer.step()
            
            train_loss += loss_sum.item()
            total_reward += np.mean(Rs)
            
        mean_train_loss = train_loss / (inference_train_size // (args.batch_size * args.num_traj))
        mean_train_reward = total_reward / (inference_train_size // (args.batch_size * args.num_traj))
        
        csgnet.eval()
        csgnet.epsilon = 0
        
        CD = 0
        for batch_idx in range(inference_test_size // args.batch_size):
            parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              args.canvas_shape)
            with torch.no_grad():
                labels = np.zeros((args.batch_size, max_len), dtype=np.int32)
                data_ = next(val_gen)
                one_hot_labels = prepare_input_op(labels, len(unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = torch.from_numpy(data_).to(device)
                test_outputs = csgnet.test([data[-1, :, 0, :, :], one_hot_labels, max_len])
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_outputs, if_just_expressions=False, if_pred_images=True)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
                CD += np.sum(chamfer(target_images, pred_images))

        CD = CD / inference_test_size
        end = time.time()
        
        utils.log_print(f"Epoch {epoch}/{epochs} =>  train loss: {mean_train_loss}, train reward: {mean_train_reward}, val cd: {CD} | {end-start}", args)

    torch.save(csgnet.state_dict(), f"{path}/best_dict.pt")
    
    return epochs+1
