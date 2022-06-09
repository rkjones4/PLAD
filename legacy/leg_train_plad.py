import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from leg_src.Models.models import ImitateJoint, ParseModelOutput
from leg_src.Models.loss import losses_joint
from leg_src.utils.generators.plad_gen import PLADGen
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
def sample_batch_modes(args):
    comb_modes = ['lest','st','ws']
    comb_weights = [args.lest_weight, args.st_weight, args.ws_weight]

    return [
        np.random.choice(
            comb_modes,
            p = comb_weights
        )
    ]

def train(csgnet, args):
    epochs = args.epochs
    path = args.infer_path
    ws_path = args.ws_save_path
    
    max_len = args.max_len
    inference_train_size = args.train_size
    inference_test_size = args.test_size

    num_traj = 1
    
    with open("legacy/terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    reinforce = Reinforce(unique_draws=unique_draw)
        
    lest_generator = PLADGen(
        f"{path}/",
        batch_size=args.batch_size,
        train_size=inference_train_size,
        mode='LEST',
        num_inst_progs=args.num_inst_progs
    )

    st_generator = PLADGen(
        f"{path}/",
        batch_size=args.batch_size,
        train_size=inference_train_size,
        mode='ST',
        num_inst_progs=args.num_inst_progs
    )

    if ws_path is not None:
        ws_generator = PLADGen(
            f"{ws_path}/",
            batch_size=args.batch_size,
            train_size=inference_train_size,
            mode='LEST',
            num_inst_progs=args.num_inst_progs
        )

    cad_generator = Generator()
    
    lest_train_gen = lest_generator.get_train_data()
    st_train_gen = st_generator.get_train_data()

    if ws_path is not None:
        ws_train_gen = ws_generator.get_train_data()
            
    val_gen = cad_generator.val_gen(
        batch_size=args.batch_size,
        path=args.data_path,
    )
        
    optimizer = optim.Adam(
        [para for para in csgnet.parameters() if para.requires_grad],
        weight_decay=args.weight_decay,
        lr=args.lr)

    best_test_loss = 1e20
    best_test_cd = 1e20
    
    torch.save(csgnet.state_dict(), f"{path}/best_dict.pt")

    patience = args.infer_patience
    num_worse = 0
        
    for epoch in range(epochs):
        start = time.time()
        losses = {}
        csgnet.train()
        b_loss = {
            'lest': [],
            'st': [],
            'ws': []
        }

        for batch_idx in range(inference_train_size //
                               (args.batch_size * num_traj)):

            optimizer.zero_grad()
            
            with torch.no_grad():
                lest_data, lest_labels = next(lest_train_gen)
                st_data, st_labels = next(st_train_gen)                
                if ws_path is not None:
                    ws_data, ws_labels = next(ws_train_gen)
                else:
                    ws_data = None
                    ws_labels = None
                    
            batch_modes = sample_batch_modes(args)

            for type_name, data, labels in [
                ('lest', lest_data, lest_labels),
                ('st', st_data, st_labels),
                ('ws', ws_data, ws_labels),
            ]:
                if type_name not in batch_modes:
                    continue            
                    
                one_hot_labels = prepare_input_op(labels, len(unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = data.to(device)
                labels = labels.to(device)
                outputs = csgnet([data, one_hot_labels, max_len])
                loss_k = ((losses_joint(outputs, labels, time_steps=max_len + 1) / (
                    max_len + 1)) / num_traj)
                loss_k.backward()
                b_loss[type_name].append(float(loss_k))
                del loss_k

            torch.nn.utils.clip_grad_norm_(csgnet.parameters(), 10)
            optimizer.step()

        b_loss = {
            k : round(torch.tensor(v).mean().item(),3) if len(v) > 0 else 0. \
            for k,v in b_loss.items()
        }
                
        csgnet.eval()
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

        if CD >= best_test_cd - args.threshold:
            num_worse += 1
        else:
            num_worse = 0
            best_test_cd = CD
            torch.save(csgnet.state_dict(), f"{path}/best_dict.pt")
            
        if num_worse >= patience:
            # load the best model and stop training
            csgnet.load_state_dict(torch.load(f"{path}/best_dict.pt"))
            return (epoch + 1) 

        end = time.time()
        utils.log_print(f"Epoch {epoch}/{epochs} =>  LEST : {b_loss['lest']}, ST : {b_loss['st']}, WS : {b_loss['ws']}, val cd: {CD} | {end-start}", args)

    return (epochs) 
