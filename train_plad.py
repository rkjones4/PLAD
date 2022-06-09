
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import time
import utils
from utils import device
import os
from tqdm import tqdm

celoss = torch.nn.CrossEntropyLoss(reduction='none')

# PLAD fine-tuning logic

class DataGen:
    def __init__(
        self,
        path,
        batch_size,
        mode,
        target_vinput
    ):
        
        self.infer_labels = torch.load(path + "infer_labels.pt").long()
        self.infer_label_weights = torch.load(path + "infer_label_weights.pt").long()
        self.infer_vinput = torch.load(path + "infer_vinput.pt").float()
        self.target_vinput = (target_vinput).float()
                
        self.train_size = self.infer_labels.shape[0]
        self.batch_size = batch_size
        self.mode = mode # AD or PL, Approx Distribution or Pseudo Labels
        
    def train_iter(self):

        ids = np.arange(self.train_size)
        np.random.shuffle(ids)
            
        for i in range(0, self.train_size, self.batch_size):
                                
            batch_labels = self.infer_labels[ids[i:i+self.batch_size]].to(device)
            batch_label_weights = self.infer_label_weights[ids[i:i+self.batch_size]].to(device)
            batch_infer_vinput = self.infer_vinput[ids[i:i+self.batch_size]]
            batch_target_vinput = self.target_vinput[ids[i:i+self.batch_size]]
                
            if self.mode == 'AD':
                batch_data = batch_infer_vinput.to(device)

            elif self.mode == 'PL':
                batch_data = batch_target_vinput.to(device)
                    
            yield (batch_data, batch_labels, batch_label_weights)

# Train generative model in the wake-sleep step
def train_gen(gen_net, real_data, args, domain):

    epochs = args.gen_epochs
    path = args.infer_path
    save_path = args.ws_save_path
    
    ad_gen = DataGen(
        f"{path}/",
        batch_size=args.batch_size,
        mode='AD',
        target_vinput=real_data.get_train_vinput()
    )

    train_gen = ad_gen.train_iter

    optimizer = optim.Adam(
        gen_net.parameters(),
        lr=args.lr
    )

    best_train_loss = 1e20

    patience = args.gen_patience
    num_worse = 0
        
    for epoch in range(epochs):
        start = time.time()
        train_loss = []
        train_cl_loss = []
        train_kl_loss = []
        
        gen_net.train()

        for vinput, progs, weights in train_gen():
            
            codes, kl_loss = gen_net.encode(vinput)
            
            preds = gen_net.decode(codes, progs)        
            
            flat_preds = preds[:,:-1,:].reshape(-1,preds.shape[2])
            flat_targets = progs[:,1:].flatten()
            flat_weights = weights[:,1:].flatten()
            
            cl_loss = (celoss(flat_preds, flat_targets) * flat_weights).sum() / (flat_weights.sum() + 1e-8)

            loss = cl_loss + (kl_loss * args.kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            train_cl_loss.append(cl_loss.item())
            train_kl_loss.append(kl_loss.item())

        mean_train_loss = torch.tensor(train_loss).mean().item()
        mean_train_kl_loss = torch.tensor(train_kl_loss).mean().item()
        mean_train_cl_loss = torch.tensor(train_cl_loss).mean().item()

        if mean_train_loss + args.threshold >= best_train_loss:
            num_worse += 1
        else:
            num_worse = 0
            best_train_loss = mean_train_loss
            
        if num_worse >= patience:
            break

        end = time.time()
        utils.log_print(f"Epoch {epoch}/{epochs} => loss: {mean_train_loss}, cl: {mean_train_cl_loss}, kl: {mean_train_kl_loss} | {end-start}", args)


    # Sample # training shapes programs from Gen Net, save to infer_labels and infer_images
    
    os.system(f'rm {save_path}infer_labels.pt')
    os.system(f'rm {save_path}infer_label_weights.pt')
    os.system(f'rm {save_path}infer_vinput.pt')

    samp_labels = torch.zeros((args.train_size, args.seq_len))
    samp_label_weights = torch.zeros((args.train_size, args.seq_len))
    samp_vinput = torch.zeros(tuple([args.train_size] + domain.get_input_shape()))
    
    gen_net.eval()

    seen = set()
    count = 0
    
    print("Sampling GEN_NET")
    # Sample from the generative model, to form next (X,Z) pairs
    pbar = tqdm(total=args.train_size)
    with torch.no_grad():
        while count < args.train_size:
            codes = torch.randn(args.batch_size, args.lat_dim, device=device)

            pred_tokens, pred_vinput = gen_net.eval_decode(
                codes   
            )
            
            for j in range(len(pred_tokens)):
                tag = tuple(pred_tokens[j].cpu().tolist())

                if tag not in seen and pred_vinput[j].sum() > 0:                    

                    samp_labels[count, :pred_tokens[j].shape[0]] = pred_tokens[j].cpu()
                    samp_label_weights[count, :pred_tokens[j].shape[0]] = 1.0
                    samp_vinput[count] = pred_vinput[j].cpu()
                    count += 1

                    pbar.update(1)
                    seen.add(tag)

                    if count == args.train_size:
                        break

    pbar.close()
            
    torch.save(samp_labels, save_path + "infer_labels.pt")
    torch.save(samp_vinput, save_path + "infer_vinput.pt")
    torch.save(samp_label_weights, save_path + "infer_label_weights.pt")    
    
    return epoch + 1

            
def sample_batch_modes(args):
    # each batch, sample data from one of the PLAD methods
    comb_modes = ['lest', 'st', 'ws']
    comb_weights = [args.lest_weight, args.st_weight, args.ws_weight]
    
    return np.random.choice(
        comb_modes,
        p = comb_weights
    )
    

def train_rec(net, gen_net, real_data, has_gen_model, args, domain):

    if has_gen_model:
        # if ws weight is greater than 0, need to train a generative model over programs
        gen_iters = train_gen(gen_net, real_data, args, domain)
    else:
        gen_iters = 0
        
    epochs = args.epochs
    path = args.infer_path
    ws_path = args.ws_save_path

    train_vinput = real_data.get_train_vinput()
    
    lest_gen = DataGen(
        f"{path}/",
        batch_size=args.batch_size,
        mode='AD',
        target_vinput=train_vinput
    )

    st_gen = DataGen(
        f"{path}/",
        batch_size=args.batch_size,
        mode='PL',
        target_vinput=train_vinput
    )

    if has_gen_model:
        ws_gen = DataGen(
            f"{ws_path}/",
            batch_size=args.batch_size,
            mode='AD',
            target_vinput=train_vinput
        )    
    
    val_gen = real_data.val_eval_iter

    opt = optim.Adam(
        net.parameters(),
        lr=args.lr
    )

    best_test_metric = domain.init_metric_val()

    torch.save(net.state_dict(), f"{path}/best_dict.pt")

    patience = args.infer_patience
    num_worse = 0
        
    for epoch in range(epochs):
        start = time.time()
        b_loss = {
            'lest': [],
            'st': [],
            'ws': []
        }
        net.train()

        lest_train_gen = lest_gen.train_iter()
        st_train_gen = st_gen.train_iter()

        if has_gen_model:
            ws_train_gen = ws_gen.train_iter()
        
        for batch_idx in range(round(args.train_size / args.batch_size)):

            with torch.no_grad():
                lest_data = next(lest_train_gen)
                st_data = next(st_train_gen)

                if has_gen_model:
                    ws_data = next(ws_train_gen)
                else:
                    ws_data = None

            batch_mode = sample_batch_modes(args)

            # Sample batch from a PLAD method, then make an MLE update to the recognition model
            for type_name, data in [
                ('lest', lest_data),
                ('st', st_data),
                ('ws', ws_data),
            ]:
                if type_name != batch_mode:
                    continue
                
                vinput, progs, weights = data
                
                codes = net.encode(vinput)
                preds = net.infer_prog(codes, progs)

                flat_preds = preds[:,:-1,:].reshape(-1,preds.shape[2])
                flat_targets = progs[:,1:].flatten()
                flat_weights = weights[:,1:].flatten()
                
                loss = (celoss(flat_preds, flat_targets) * flat_weights).sum() / (flat_weights.sum() + 1e-8)

                opt.zero_grad()
                loss.backward()
                opt.step()
            
                b_loss[type_name].append(loss.item())


        b_loss = {
            k : round(torch.tensor(v).mean().item(),3) if len(v) > 0 else 0. \
            for k,v in b_loss.items()
        }
        
        net.eval()
        metric = []
        
        with torch.no_grad():
            for vinput in val_gen():
                _, _, pred_metric = net.eval_infer_progs(vinput, args.es_beams)
                metric += pred_metric

        METRIC = torch.tensor(metric).mean().item()
        
        ## EVAL

        if not domain.should_save(METRIC, best_test_metric, args.threshold):
            num_worse += 1
        else:
            num_worse = 0
            best_test_metric = METRIC
            torch.save(net.state_dict(), f"{path}/best_dict.pt")

        # early stopping on validation set 
        if num_worse >= patience:
            # load the best model and stop training
            net.load_state_dict(torch.load(f"{path}/best_dict.pt"))
            return epoch + 1, gen_iters

        end = time.time()
        utils.log_print(f"Epoch {epoch}/{epochs} =>  LEST : {b_loss['lest']}, ST : {b_loss['st']}, WS : {b_loss['ws']} | val metric: {METRIC} | {end-start}", args)

    return epochs, gen_iters
