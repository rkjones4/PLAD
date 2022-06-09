import os, sys
sys.path.append('legacy')
import torch
from leg_src.Models.models import Encoder, ImitateJoint, VAE
from leg_infer import infer_programs
import leg_train_rl, leg_train_ws, leg_train_plad
import time
import utils
import json
from utils import device
import matplotlib.pyplot as plt

# Special case CSG 2D for fair-comparison against CSGNet

class CSG2D_DOMAIN:    
    def legacy_fine_tune(self):
        fine_tune()

DEF_ARGS = [
    # Need to set
    ('-en', '--exp_name', None,  str),
    ('-ftm', '--ft_mode', None, str),

    ('-o', '--outpath', 'model_output',  str),
    # Defs
    ('-pl', '--preload_model', 'False', str),
    ('-plm', '--pretrain_modelpath', 'model_output/CSG2D_SP/models/infer_net.pt', str),
    
    ('-ntraj', '--num_traj', 1, int),
    ('-e', '--epochs', 1000, int),
    ('-bs', '--batch_size', 100, int),
    ('-hd', '--hidden_size', 2048, int),
    ('-is', '--input_size', 2048, int),    
    ('-lr', '--lr', 0.001, float),
    ('-ed', '--encoder_drop', 0.2, float),
    ('-dd', '--dropout', 0.2, float),
    ('-wd', '--weight_decay', 0.0, float),        
    ('-eps', '--eps', 1, float),
    ('-ml', '--max_len', 13, int),
    ('-ts', '--train_size', 10000, int),
    ('-es', '--test_size', 3000, int),    
    ('-dp', '--data_path', 'data/csgnet_cad/cad.h5', str),    
    
    ('-rd', '--rd_seed', 42, float),
    ('-npixels', '--num_pixels', 64, int),

    ('-mi', '--max_iters', 100000, int),
    ('-infp', '--infer_patience', 10, int),
    ('-itrp', '--iter_patience', 1000, int),
    ('-thr', '--threshold', 0.005, float),

    ('-ltd', '--lat_dim', 128, int),
    ('-ve', '--vae_epochs', 100, int),
    ('-vp', '--vae_patience', 10, int),
    ('-kl', '--kl_weight', 1.0, float),

    ('-tm', '--train_mode', 'fine-tune', str),
    ('-sm', '--sample_fn', 'beam', str), 
    ('-nip', '--num_inst_progs', 1, int),
    ('-bw', '--beam_width', 10, int),
    ('-km', '--keep_mode', 'at', str), 

    ('-lest_w', '--lest_weight', 0.0, float),
    ('-st_w', '--st_weight', 0.0, float),
    ('-ws_w', '--ws_weight', 0.0, float),
    ('-rl_w', '--rl_weight', None, float),
]

def get_args():
    args = utils.getArgs(DEF_ARGS)
        
    args.infer_path = f"model_output/{args.exp_name}/train_out/"
    args.ws_save_path = f"model_output/{args.exp_name}/ws_out/"    
    
    if 'LEST' in args.ft_mode:
        args.lest_weight = 1.0

    if 'ST' in args.ft_mode:
        args.st_weight = 1.0

    if 'WS' in args.ft_mode:
        args.ws_weight = 1.0
        
    norm = args.lest_weight + args.st_weight  + args.ws_weight

    if norm > 0:
                
        args.lest_weight = args.lest_weight / norm
        args.st_weight = args.st_weight / norm
        args.ws_weight = args.ws_weight / norm        
            
        
    args = args
    utils.init_exp_model_run(args)

    return args

"""
Get initial pretrained CSGNet inference network
"""
def get_csgnet(config):

    # Encoder
    encoder_net = Encoder(config.encoder_drop)
    encoder_net = encoder_net.to(device)

    imitate_net = ImitateJoint(
        hd_sz=config.hidden_size,
        input_size=config.input_size,
        encoder=encoder_net,
        num_draws=400,
        canvas_shape=config.canvas_shape)
    imitate_net = imitate_net.to(device)

    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath, map_location=device)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    return imitate_net


def get_vae(config):

    # Encoder
    encoder_net = Encoder(config.encoder_drop)
    encoder_net = encoder_net.to(device)

    vae = VAE(
        hd_sz=config.hidden_size,
        input_size=config.input_size,
        encoder=encoder_net,
        num_draws=400,
        canvas_shape=config.canvas_shape,
        lat_dim=config.lat_dim
    )
    vae = vae.to(device)

    return vae

def fine_tune():
    args = get_args()
    args.canvas_shape = [args.num_pixels, args.num_pixels]
    csgnet = get_csgnet(args)
        
    inf_epochs = 0
    
    epochs = []
    res = {
        'train': [],
        'val': [],
        'test': [],        
    }

    if 'WS' in args.ft_mode:
        vae = get_vae(args)
        vae_epochs = []
        _vae_epochs = 0
        args.ws_save_path = args.infer_path
        

    if args.ft_mode != 'RL':
        utils.log_print(f"PLAD Weights :  {args.lest_weight} , {args.st_weight} , {args.ws_weight}", args)
        
    if args.ws_weight > 0:
        vae = get_vae(args)
        vae_epochs = []
        _vae_epochs = 0
        args.ws_save_path = f'model_output/{args.exp_name}/ws_out/'
        os.system(f'mkdir {args.ws_save_path} > /dev/null 2>&1')
    else:
        args.ws_save_path = None
            
    Round = 0

    best_val = 1e8
    best_epoch = 0    
    
    os.system(f'rm -f {args.infer_path}/*.pt')
    
    while inf_epochs < args.max_iters:
        utils.log_print(f"ROUND {Round} (Inf Epochs: {inf_epochs})", args)
        
        iter_res = infer_programs(csgnet, args)

        for part, part_cd in iter_res.items():
            res[part].append(part_cd)

        epochs.append(inf_epochs)

        eres = {k:v for k,v in res.items()}
        eres['epochs'] = epochs

        if 'WS' in args.ft_mode:
            vae_epochs.append(_vae_epochs)
            eres['vae_epochs'] = vae_epochs
        
        json.dump(eres, open(f"model_output/{args.exp_name}/res.json" ,'w'))
        del eres
        
        make_plots(res, epochs, args, 'CD')
        
        if (iter_res['val'] + args.threshold) < best_val:
            utils.log_print("Replacing best model", args)
            best_val = iter_res['val']
            best_epoch = inf_epochs                    
            torch.save(csgnet.state_dict(), f"model_output/{args.exp_name}/inf_net.pt")
            
        if inf_epochs - best_epoch > args.iter_patience:
            utils.log_print("Stopping early", args)
            break                    
        
        if args.ft_mode == 'RL':
            inf_epochs += leg_train_rl.train(csgnet, args)
    
        else:
            if args.ws_weight > 0:
                _vae_epochs += leg_train_ws.train(vae, args)

            inf_epochs += leg_train_plad.train(csgnet, args)
                
        torch.save(csgnet.state_dict(), f"{args.infer_path}/model.pt")
        
        Round += 1

    os.system(f'rm -f {args.infer_path}/*.pt')


def make_plots(res, epochs, args, name):
    plt.clf()
    for key, vals in res.items():
        plt.plot(
            epochs,
            vals,
            label = key
        )
    plt.legend()
    plt.grid()
    plt.savefig(f'model_output/{args.exp_name}/plots/{name}.png')
