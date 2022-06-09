import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from leg_src.Models.models import ImitateJoint, ParseModelOutput
from leg_src.Models.loss import losses_joint
from leg_src.utils.generators.plad_gen import PLADGen
from leg_src.utils.train_utils import prepare_input_op, chamfer, beams_parser, image_from_expressions
from leg_src.utils.generators.shapenet_generater import Generator
import time
import utils
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import device


def train(vae, args):
    epochs = args.vae_epochs
    path = args.infer_path
    save_path = args.ws_save_path
    max_len = args.max_len
    inference_train_size = args.train_size

    with open("legacy/terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              args.canvas_shape)
        
    generator = PLADGen(
        f"{path}/",
        batch_size=args.batch_size,
        train_size=inference_train_size,
        mode='LEST',
        num_inst_progs=args.num_inst_progs
    )

    train_gen = generator.get_train_data()

    optimizer = optim.Adam(
        [para for para in vae.parameters() if para.requires_grad],
        weight_decay=args.weight_decay,
        lr=args.lr)

    best_train_loss = 1e20

    patience = args.vae_patience
    num_worse = 0
        
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        train_cl_loss = 0
        train_kl_loss = 0
        
        vae.train()
        for batch_idx in range(inference_train_size // args.batch_size):            

            data, labels = next(train_gen)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
            data = data.to(device)
            labels = labels.to(device)

            codes, kl_loss = vae.encode(data)
            
            outputs = vae.decode(codes, one_hot_labels, max_len)
                                
            cl_loss = (losses_joint(outputs, labels, time_steps=max_len + 1) / (
                max_len + 1))

            loss = cl_loss + (kl_loss * args.kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_cl_loss += cl_loss.item()
            train_kl_loss += kl_loss.item()
            
        mean_train_loss = train_loss / (inference_train_size // (args.batch_size))
        mean_train_kl_loss = train_kl_loss / (inference_train_size // (args.batch_size))
        mean_train_cl_loss = train_cl_loss / (inference_train_size // (args.batch_size))

        if mean_train_loss + args.threshold >= best_train_loss:
            num_worse += 1
        else:
            num_worse = 0
            best_train_loss = mean_train_loss
            
        if num_worse >= patience:
            break

        end = time.time()
        utils.log_print(f"Epoch {epoch}/{epochs} => loss: {mean_train_loss}, cl: {mean_train_cl_loss}, kl: {mean_train_kl_loss} | {end-start}", args)


    # Sample # training shapes programs from VAE, save to infer_labels and infer_images
    
    os.system(f'rm {save_path}infer_labels.pt')
    os.system(f'rm {save_path}infer_images.pt')
    os.system(f'rm {save_path}target_images.pt')

    samp_labels = []
    samp_images = []

    vae.eval()

    seen = set()

    print("Sampling VAE")
    pbar = tqdm(total=args.train_size)
    with torch.no_grad():
        while len(samp_labels) < args.train_size:
            codes = torch.randn(args.batch_size, args.lat_dim, device=device)

            ucodes = vae.up_ll(codes)
            
            labels = np.zeros((args.batch_size, max_len), dtype=np.int32)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
            
            all_beams = vae.csgnet.ws_eval_decode(
                ucodes,
                one_hot_labels,
                max_len
            )

            beam_labels = beams_parser(
                all_beams, args.batch_size, beam_width=1
            )

            beam_labels_numpy = np.zeros(
                    (args.batch_size, max_len), dtype=np.int32)
                
            for i in range(args.batch_size):
                beam_labels_numpy[i] = beam_labels[i]
            
            # find expression from these predicted beam labels
            expressions = [""] * args.batch_size
            for i in range(args.batch_size):
                for j in range(max_len):
                    expressions[i] += unique_draw[beam_labels_numpy[i, j]]
                
            for index, prog in enumerate(expressions):
                expressions[index] = prog.split("$")[0]
            
            pred_images = image_from_expressions(parser, expressions)

            for j in range(args.batch_size):
                if expressions[j] not in seen and pred_images[j].sum() > 0:
                    samp_labels.append(beam_labels[j])
                    samp_images.append(pred_images[j])
                    pbar.update(1)
                    seen.add(expressions[j])

    pbar.close()

    samp_labels = np.stack(samp_labels, axis=0)[:args.train_size]
    samp_images = np.stack(samp_images, axis=0)[:args.train_size]

    samp_labels = samp_labels.reshape(args.train_size, 1, -1)
    samp_images = samp_images.reshape(args.train_size, 1, samp_images.shape[1], samp_images.shape[2])

    torch.save(samp_labels, save_path + "infer_labels.pt")
    torch.save(samp_images, save_path + "infer_images.pt")
    torch.save(np.zeros((args.train_size, samp_images.shape[2], samp_images.shape[3])), save_path + "target_images.pt")   
    
    return epoch + 1
