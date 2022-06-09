import data.shapenet.load_data as load_data
from models.models_sa import ProgInfNet
import executors.ex_sa as ex
import train_plad
import train_rl
import sys
from infer import infer_programs
from utils import device
import utils
import torch
import numpy as np
import random
from tqdm import tqdm

# Config file for the shapeAssembly domain
# See csg3d file for comments

CAD_CATS = [
    (cat, f'data/shapenet/{cat}_vox.hdf5') for cat in (
        'bench',
        'chair',
        'couch',
        'table',
    )
]

TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'batch_count'),
    ('Accuracy', 'corr', 'total')
]

EVAL_LOG_INFO = [    
    ('IoU', 'iou', 'count'),
    ('Accuracy', 'corr', 'total'),
] 


FT_ARGS = [
    ('-en', '--exp_name', None,  str),
    ('-ftm', '--ft_mode', None, str),
    
    ('-rd', '--rd_seed', 42,  int),
    ('-lr', '--lr', 0.0005,  float),
    ('-o', '--outpath', 'model_output',  str),

    ('-dp', '--dropout', .1, float),
    ('-esp', '--es_patience', 10,  int),
    ('-est', '--es_threshold', 0.001,  float),
    ('-esm', '--es_metric', 'IoU',  str),        
    ('-km', '--keep_mode', 'at', str), # at  for all-time
    ('-ts', '--train_size', 10000, int),
    ('-es', '--eval_size', 1000, int), 
    ('-pmp', '--pretrain_modelpath', 'model_output/SA_SP/models/infer_net.pt', str),

    ('-sl', '--seq_len', 100, int),
    ('-nl', '--num_layers', 8, int),
    ('-nh', '--num_heads', 16, int),    
    ('-hd', '--hidden_dim', 256, int),

    ('-evp', '--eval_per', 1, int),
    ('-prp', '--print_per', 1, int),
    
    ('-mi', '--max_iters', 10000, int),
    ('-infp', '--infer_patience', 10, int),
    ('-itrp', '--iter_patience', 100, int),
    ('-thr', '--threshold', 0.001, float),

    ('-c', '--categories', 'chair,couch,table,bench', str),
    ('-bs', '--batch_size', 100, int),
    ('-ebs', '--eval_batch_size', 100, int),
        
    ('-beams', '--beams', 10, int),
    ('-inb', '--infer_beams', 5, int),
    ('-esb', '--es_beams', 3, int),
    ('-e', '--epochs', 100, int),

    ('-nt', '--num_traj', 10, int),

    # WS

    ('-ltd', '--lat_dim', 128, int),
    ('-ge', '--gen_epochs', 100, int),
    ('-gp', '--gen_patience', 10, int),
    ('-kl', '--kl_weight', 1.0, float),

    # COMB
    
    ('-lest_w', '--lest_weight', 0., float),
    ('-st_w', '--st_weight', 0., float),
    ('-ws_w', '--ws_weight', 0., float),
]

PT_ARGS = [    
    # Set per run
    ('-en', '--exp_name', None,  str),        
    
    ('-rd', '--rd_seed', 42,  int),    

    ('-ni', '--num_iters', 1000,  int), # batches    
    ('-lr', '--lr', 0.0005,  float),
    ('-o', '--outpath', 'model_output',  str),
    
    ('-nw', '--num_write', 10, int),    
    ('-dp', '--dropout', .1, float),

    ('-esp', '--es_patience', 10,  int),
    ('-est', '--es_threshold', 0.001,  float),
    ('-esm', '--es_metric', 'IoU',  str),

    ('-beams', '--beams', 10, int),
    ('-mip', '--min_prims', 2, int),
    ('-map', '--max_prims', 8, int),
    
    ('-sl', '--seq_len', 100, int),
    ('-nl', '--num_layers', 8, int),
    ('-nh', '--num_heads', 16, int),    
    ('-hd', '--hidden_dim', 256, int),
    ('-bs', '--batch_size', 400, int),
    
    ('-ts', '--train_size', 2000000, int),
    ('-es', '--eval_size', 1000, int),
    ('-evp', '--eval_per', 1, int),
    ('-prp', '--print_per', 1, int),
]

def get_rec_net(args, ex, do_load=True):

    net = ProgInfNet(
        ex,
        args.hidden_dim,
        args.seq_len,
        args.dropout,
        device,
        args.batch_size,
        args.num_layers,
        args.num_heads,
        args.beams
    )

    if not do_load:
        net.to(device)
        return net
    
    model_path = args.pretrain_modelpath
        
    net.load_state_dict(
        torch.load(model_path)
    )
    net.to(device)

    return net

def get_gen_model(args, ex):

    net = ProgInfNet(
        ex,
        args.hidden_dim,
        args.seq_len,
        args.dropout,
        device,
        args.batch_size,
        args.num_layers,
        args.num_heads,
        args.beams
    )

    vae = net.add_vae(args.lat_dim)
        
    vae = vae.to(device)

    return vae

class SynthDataset:
    def __init__(
        self, args, size, eval_size
    ):        
        self.mode = 'train'

        self.size = size
        self.eval_size = eval_size
        self.batch_size = args.batch_size

        self.seq_len = args.seq_len
        self.max_prims = args.max_prims
        self.min_prims = args.min_prims
                            
        self.tokens = []
        self.progs = []
        
        data = ex.prog_random_sample(
            self.size, self.seq_len, self.max_prims,
            self.min_prims, ret_voxels = False
        )

        for prog in data[:self.size]:
            self.tokens.append(np.array([ex.T2I[t] for t in prog]))
            self.progs.append(prog)
            
    def __iter__(self):

        if self.mode == 'train':
            yield from self.train_static_iter()
                            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def train_static_iter(self):
        
        inds = list(range(len(self.progs)))
        random.shuffle(inds)

        while len(inds) > 0:
            binds = inds[:self.batch_size]
            inds = inds[self.batch_size:]
            
            batch_voxels = torch.zeros(
                len(binds), ex.DIM, ex.DIM, ex.DIM, device=device
            ).float()
            batch_progs = torch.zeros(len(binds), self.seq_len, device=device).long()
            batch_weights = torch.zeros(len(binds), self.seq_len, device=device).float()
            
            with torch.no_grad():                
                for i, ind in enumerate(binds):

                    prog = self.progs[ind]
                    tokens = self.tokens[ind]                    
                    voxels = ex.execute(' '.join(prog))
            
                    batch_voxels[i] = torch.from_numpy(voxels).float().to(device)
                    bprog = torch.from_numpy(tokens).long().to(device)
                    batch_progs[i,:bprog.shape[0]] = bprog
                    batch_weights[i,:bprog.shape[0]] = 1.0
                
            yield batch_voxels, batch_progs, batch_weights            
    
    def eval_iter(self):
        for i in tqdm(range(self.eval_size)):
            prog = self.progs[i]
            tokens = self.tokens[i]
            voxels = ex.execute(' '.join(prog))
            yield torch.from_numpy(voxels).float().to(device), \
                torch.from_numpy(tokens).long()

class CADDataset:
    def __init__(
        self, args
    ):        

        self.eval_batch_size = args.eval_batch_size
        self.batch_size = args.batch_size
        
        self.cats = []
        self.vinput = []

        valid_cats = args.categories.split(',')
        print("Loading CAD data")
        for ci, (cat, cat_file) in enumerate(CAD_CATS):
            if cat not in valid_cats:
                continue
            
            with torch.no_grad():            
                cat_vinput = load_data.get_voxels(cat_file, args.train_size + args.eval_size + args.eval_size)
                self.vinput += cat_vinput
                self.cats += [ci] * len(cat_vinput)
                utils.log_print(f'Cat {cat} | CI {ci} | Num {len(cat_vinput)}', args)
                
        self.cats = torch.tensor(self.cats)
        self.vinput = torch.stack(self.vinput, dim = 0)

        assert self.vinput.shape[0] > args.train_size + args.eval_size + args.eval_size
        
        inds = torch.randperm(self.vinput.shape[0])

        self.train_inds = inds[:args.train_size]
        self.val_inds = inds[ args.train_size : args.train_size + args.eval_size]
        self.test_inds = inds[ args.train_size+args.eval_size : args.train_size+ args.eval_size + args.eval_size]

        print(f'Train {self.train_inds.shape[0]} | Val {self.val_inds.shape[0]} | Test {self.test_inds.shape[0]}')

        torch.save(self.train_inds, f'model_output/{args.exp_name}/train_inds.pt')
        torch.save(self.val_inds, f'model_output/{args.exp_name}/val_inds.pt')
        torch.save(self.test_inds, f'model_output/{args.exp_name}/test_inds.pt')

    def train_rl_iter(self):
        while True:
            inds = torch.randperm(self.train_inds.shape[0])

            for start in range(
                    0, inds.shape[0], self.batch_size
            ):
                sinds = inds[start:start+self.batch_size]
                binds = self.train_inds[sinds]
                yield self.vinput[binds].float().to(device)                        
            
    def get_train_vinput(self):
        return self.vinput[self.train_inds]
        
    def train_eval_iter(self):
        inds = self.train_inds
        yield from self.eval_iter(inds)

    def val_eval_iter(self):
        inds = self.val_inds
        yield from self.eval_iter(inds)

    def test_eval_iter(self):
        inds = self.test_inds
        yield from self.eval_iter(inds)

    def eval_iter(self, inds):
        for start in range(
            0, inds.shape[0], self.eval_batch_size
        ):
            binds = inds[start:start+self.eval_batch_size]
            yield self.vinput[binds].float().to(device)            


class SA_DOMAIN:
    def __init__(self):        
        self.metric_name = 'IOU'
        self.executor = ex
        
    def load_pretrain_net(self):
        return get_rec_net(self.args, self.executor)

    def load_new_net(self):
        return get_rec_net(self.args, self.executor, False)
    
    def load_real_data(self):
        return CADDataset(self.args)

    def create_gen_model(self,):
        return get_gen_model(self.args, self.executor)

    def should_save(self, cur_val, best_val, thresh):
        if self.comp_metric(cur_val - thresh, best_val):
            return True
        else:
            return False

    def init_metric_val(self):
        return 0.
        
    def comp_metric(self, a, b):
        if a > b:
            return True
        else:
            return False
        
    def init_rl_run(self):
        self.reinforce = train_rl.Reinforce()

    def infer_programs(self, net, real_data):
        return infer_programs(net, real_data, self.args, self)

    def train_rl(self, net, real_data):
        return train_rl.train_rec(net, real_data, self.args, self)

    def train_plad(self, rec_net, gen_net, real_data, has_gen_model):
        ie, ge = train_plad.train_rec(
            rec_net, gen_net, real_data, has_gen_model, self.args, self
        )
        return ie, ge
        
    def get_input_shape(self):
        return self.executor.get_input_shape()

    def get_synth_datasets(self, args):
        print("Loading Synth Data")
        train_loader = SynthDataset(args, args.train_size, args.eval_size)
        val_loader = SynthDataset(args, args.eval_size, args.eval_size)

        return train_loader, val_loader
        
    def get_pt_args(self):
        args = utils.getArgs(PT_ARGS)
        self.args = args
        self.TRAIN_LOG_INFO = TRAIN_LOG_INFO
        self.EVAL_LOG_INFO = EVAL_LOG_INFO
        utils.init_pretrain_run(args)
        
        return args
        
    def get_ft_args(self):
        args = utils.getArgs(FT_ARGS)
        
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
            
        if args.ft_mode == 'RL':
            params = ' '.join(sys.argv[1:])

            # Params for RL matching CSGNet / Adjusting for memory requirements
        
            if not ('--learning_rate ' in params or '-lr ' in params):
                print("Set RL LR to DEF 01")
                args.lr = 0.01
            
            if not ('--epochs ' in params or '-e ' in params):
                print("Set RL EP to DEF 2")
                args.epochs = 2

            if not ('--batch_size ' in params or '-b ' in params):
                print("Set RL Batch Size to DEF 4")
                args.batch_size = 4
                
        self.args = args
        utils.init_exp_model_run(self.args)

        return args
