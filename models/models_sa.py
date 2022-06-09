import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.model_utils as mu
    
# Recognition model for CSG SA domain
# See comments in models_csg3d, main difference is use of END_TOKEN instead of bqc to determine when to stop auto-regressive decoding

class ProgInfNet(nn.Module):    
    def __init__(
        self,
        ex,
        hidden_dim,
        seq_len,
        dropout,
        device,
        batch_size,
        num_layers,
        num_heads,
        beams
    ):
        super(ProgInfNet, self).__init__()

        self.ex = ex
        self.beams = beams

        if self.beams == 0:
            self.eval_mode = 'argmax'
        else:
            self.eval_mode = 'beam'
        
        self.device= device

        self.mp = 8
        
        self.ms = seq_len

        self.nl = num_layers
        self.nh = num_heads
                
        self.bs = batch_size
        self.dropout = dropout
        
        self.es = len(ex.TOKENS)
        self.hd = hidden_dim 
                
        self.token_enc_net = nn.Embedding(self.es, self.hd)        
        self.token_head = mu.SDMLP(self.hd, self.es, self.dropout)

        self.pos_enc = nn.Embedding(self.ms+self.mp, self.hd)
        self.pos_arange = torch.arange(self.ms+self.mp).unsqueeze(0)
                    
        self.attn_mask = self.generate_attn_mask()

        self.attn_layers = nn.ModuleList([mu.AttnLayer(self.nh, self.hd, self.dropout) for _ in range(self.nl)])

        self.cnn = mu.vox3DCNN(hidden_dim, dropout)

        self.epsilon = 0.
                        
            
    def add_vae(self, lat_dim):
        return mu.VAE(self, lat_dim)
        
    def generate_attn_mask(self):
        return mu._generate_attn_mask(self)

    def generate_key_mask(self, num):
        return mu._generate_key_mask(self, num)

    def encode(self, voxels):
        return self.cnn(voxels)
    
    # prims: B X HD , floats
    # seq: B X MS , long

    def infer_prog(self, codes, seq):
        return mu._infer_prog(self, codes, seq)
            
    def eval_infer_progs(self, voxels, beam_size = None):
        
        if beam_size is None:
            beams = self.beams
        else:
            beams = beam_size
            
        batch = voxels.shape[0]

        codes = self.encode(voxels)        
        
        bseqs = torch.zeros(batch * beams, self.ms, device=self.device).long()
        bseqs[:,0] = self.ex.T2I[self.ex.START_TOKEN]    
        
        blls = torch.zeros(batch, beams, device=self.device)        
        blls[:,1:] += mu.MIN_LL_PEN
        blls = blls.flatten()

        # [batch, beam, O]

        bcodes = codes.view(
            codes.shape[0], 1, codes.shape[1], codes.shape[2]
        ).repeat(1, beams, 1, 1).view(
            beams * batch, codes.shape[1], codes.shape[2]
        )
        
        fin_progs = {i:[] for i in range(batch)}

        END_TOKEN = self.ex.T2I['END']
        
        for ti in range(0, self.ms-1):
            
            bpreds = self.infer_prog(bcodes, bseqs)[:,ti,:]
            
            bdist = torch.log(torch.softmax(bpreds, dim = 1) + 1e-8)            
            
            beam_liks, beam_choices = torch.topk(bdist, beams)
            
            next_liks = (beam_liks + blls.view(-1, 1)).view(batch, -1)

            E_ll, E_ranked_beams = torch.sort(next_liks,1,True)

            blls = E_ll[:,:beams].flatten()

            ranked_beams = E_ranked_beams[:,:beams]

            R_beam_choices = beam_choices.view(batch, -1)

            nt = torch.gather(R_beam_choices,1,ranked_beams).flatten()

            old_index = (torch.div(ranked_beams, beams, rounding_mode='floor') + (torch.arange(batch, device=self.device) * beams).view(-1, 1)).flatten()
            #old_index = (torch.floor_divide(ranked_beams, beams) + (torch.arange(batch, device=device) * beams).view(-1, 1)).flatten()
            
            bseqs  = bseqs[old_index].clone()
            bseqs[:, ti+1] = nt

            bcodes = bcodes[old_index]
                                
            fin_inds = (nt == END_TOKEN).nonzero().flatten().tolist()
            
            for i in fin_inds:                
                                                                
                if blls[i] > mu.MIN_LL_THRESH:
                    fin_progs[i // beams].append((
                        blls[i].item(),
                        bseqs[i,:ti+2]
                    ))
                    
                blls[i] += mu.MIN_LL_PEN
            
        best_progs = []
        best_ious = []
        best_voxels = []
        
        for i, fp in fin_progs.items():
            
            fp.sort(reverse=True, key=lambda a: a[0])

            c = 0
            best_iou = 0
            best_tokens = torch.tensor([])
            best_vox = torch.zeros(self.ex.get_input_shape()).float()
            
            B_voxels = voxels[i] > 0

            for _, tokens in fp:
                if c >= beams:
                    break

                try:
                
                    prog = ' '.join([self.ex.I2T[p.item()] for p in tokens])
                    pred_voxels = torch.from_numpy(self.ex.execute(prog)).to(B_voxels.device)
                    
                    Bpred_voxels = pred_voxels > 0
                    inter = (B_voxels & Bpred_voxels).sum().item()
                    union = (B_voxels | Bpred_voxels).sum().item()
                    _iou = inter * 1. / union
                
                    c += 1
                
                    if _iou > best_iou:
                        best_iou = _iou
                        best_tokens = tokens
                        best_vox = pred_voxels.cpu()
                        
                except Exception as e:
                    pass

            best_progs.append(best_tokens)
            best_ious.append(best_iou)
            best_voxels.append(best_vox)

        return best_progs, best_voxels, best_ious

    
    def ws_sample(self, codes):
                    
        batch = codes.shape[0]
        
        bseqs = torch.zeros(batch, self.ms, device=self.device).long()
        bseqs[:,0] = self.ex.T2I[self.ex.START_TOKEN]    
        
        blls = torch.zeros(batch, device=self.device)        

        END_TOKEN = ex.T2I['END']
    
        # [batch, beam, O]

        bcodes = codes
        
        fin_progs = []

        old_index = torch.arange(batch,device=self.device).long()
        
        for ti in range(0, self.ms-1):
            
            bpreds = self.infer_prog(bcodes, bseqs)[:,ti,:]
            
            bdist = torch.softmax(bpreds, dim = 1)

            nt = torch.distributions.categorical.Categorical(bdist).sample()        

            bseqs[:, ti+1] = nt
            
            fin_inds = (nt == END_TOKEN).nonzero().flatten().tolist()
            
            for i in fin_inds:                
                                                                
                if blls[i] > mu.MIN_LL_THRESH:
                    fin_progs.append((
                        bseqs[i,:ti+2]
                    ))
                    
                blls[i] += mu.MIN_LL_PEN
            
        progs = []
        voxels = []
        
        for tokens in fin_progs:
                            
            try:
                prog = ' '.join([self.ex.I2T[p.item()] for p in tokens])
                pred_voxels = torch.from_numpy(self.ex.execute(prog))
                
                progs.append(tokens)
                voxels.append(pred_voxels)
                
            except Exception as e:
                pass

        return progs, voxels
    
    def rl_fwd(self, voxels):
        return mu._rl_fwd(self, voxels)
                
    def generate_rewards(self, samples, voxels, power = 1.):
        len_progs = []
        R = []

        Tsamples = torch.stack(samples,dim=0).T

        END_TOKEN = self.ex.T2I['END']
        
        for samp, target in zip(Tsamples, voxels):

            tokens = [self.ex.T2I['START']]            

            l = 0

            while l < samp.shape[0]:
            

                nt = samp[l].item()                               
                tokens.append(nt)
                l += 1

                if nt == END_TOKEN:
                    break

            B_voxels = target > 0
            
            try:
                prog = ' '.join([self.ex.I2T[p] for p in tokens])
                pred_voxels = torch.from_numpy(self.ex.execute(prog)).to(B_voxels.device)
                Bpred_voxels = pred_voxels > 0
                inter = (B_voxels & Bpred_voxels).sum().item()
                union = (B_voxels | Bpred_voxels).sum().item()
                _R = inter * 1. / union
                
            except Exception as e:
                _R = 0.
            
            R.append(_R ** power)
            len_progs.append(l)
        
        R = torch.tensor(R)    
        return R, len_progs
