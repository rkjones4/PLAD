import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Util functions for recongition network models

# penalties for violating different constraints
MIN_LL_THRESH = -1000
MIN_LL_PEN = -10000
MASK_LL_PEN = -100000.

# dropout MLP

class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)

# small dropout MLP
class SDMLP(nn.Module):
    def __init__(self, ind, odim, DP):
        super(SDMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, odim)
        self.l2 = nn.Linear(odim, odim)
        self.d1 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.leaky_relu(self.l1(x), 0.2))
        return self.l2(x)

# 3D voxel CNN encoder
class vox3DCNN(nn.Module):
    def __init__(self, code_size, drop):

        super(vox3DCNN, self).__init__()
                                                
        # Encoder architecture
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b3 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b4 = nn.BatchNorm3d(num_features=256)

        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(drop),
            self.b4,
        )
        

        self.ll = DMLP(256, 256, 256, code_size, drop)
            
    def forward(self, x):
        x1 = x.view(-1, 1, 32, 32, 32)
        x2 = self._encoder(x1)
        
        x2 = x2.view(-1, 256, 8)            
        x2 = x2.transpose(1, 2)
                        
        return self.ll(x2)


######## TRANSFORMER

class AttnLayer(nn.Module):
    def __init__(self, nh, hd, dropout):
        super(AttnLayer, self).__init__()
        self.nh = nh
        self.hd = hd

        self.self_attn = torch.nn.MultiheadAttention(self.hd, self.nh)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)        

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
                
    def forward(self, _src, attn_mask, key_padding_mask):
        
        src = _src.transpose(0, 1)
            
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask
        )[0]

        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(src), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)

        return src.transpose(0, 1)

# VAE sampling layer
class Sampler(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, hidden_size)
        self.mlp2var = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        
        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)

        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)

        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return eps.mul(std).add_(mu), -kld.sum()


# VAE with encoder, decoder, and sampling layer
class VAE(nn.Module):
    def __init__(
        self,
        inf_net,
        lat_dim
    ):
        super(VAE, self).__init__()
        self.net = inf_net
        self.sampler = Sampler(self.net.hd * self.net.mp, lat_dim)
        self.up_ll = nn.Linear(lat_dim, self.net.hd * self.net.mp)
        
    def encode(self, voxels):
        codes = self.net.encode(voxels).view(voxels.shape[0], -1)        
        scodes, kl_loss = self.sampler(codes)        
        return scodes, kl_loss

    def decode(self, codes, progs):
        ucodes = self.up_ll(codes).view(progs.shape[0], self.net.mp, self.net.hd)        
        return self.net.infer_prog(ucodes, progs)

    def eval_decode(self, codes):
        ucodes = self.up_ll(codes).view(codes.shape[0], self.net.mp, self.net.hd)        
        return self.net.ws_sample(ucodes)
    
# generate attention mask for transformer auto-regressive training
# first mp spaces have fully connected attention, as they are the priming sequence of visual encoding
def _generate_attn_mask(net):
    sz = net.mp + net.ms
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).T
    mask[:net.mp, :net.mp] = 0.
    return mask

# generate key mask for transformer auto-regressive training
def _generate_key_mask(net, num):
    sz = net.mp + net.ms
    mask = torch.zeros(num, sz).bool()
    return mask

# main forward process of transformer, encode tokens, add PE, run through attention, predict tokens with MLP
def _infer_prog(net, codes, seq):
    token_encs = net.token_enc_net(seq).view(-1, net.ms, net.hd)
                                            
    out = torch.cat((codes.view(codes.shape[0], net.mp, net.hd), token_encs), dim = 1)        
    out += net.pos_enc(net.pos_arange.repeat(codes.shape[0], 1).to(net.device))
        
    attn_mask = net.attn_mask.to(net.device)
    key_mask = net.generate_key_mask(codes.shape[0]).to(net.device)
        
    for attn_layer in net.attn_layers:        
        out = attn_layer(out, attn_mask, key_mask)
        
    seq_out = out[:,net.mp:,:]

    token_out = net.token_head(seq_out)
        
    return token_out

# rl helper function, for forward process
def _rl_infer_prog(net, codes, seq):
    batch_size = seq.shape[0]
        
    token_encs = net.token_enc_net(seq).view(batch_size, -1, net.hd)
                                            
    out = torch.cat((codes.view(codes.shape[0], net.mp, net.hd), token_encs), dim = 1)
    out += net.pos_enc(net.pos_arange.repeat(codes.shape[0], 1).to(net.device))[:,:out.shape[1]]
        
    attn_mask = net.attn_mask.to(net.device)
    key_mask = net.generate_key_mask(codes.shape[0]).to(net.device)
        
    for attn_layer in net.attn_layers:        
        out = attn_layer(out, attn_mask[:out.shape[1],:out.shape[1]], key_mask[:,:out.shape[1]])
        
    seq_out = out[:,net.mp:,:]
    
    token_out = net.token_head(seq_out)
        
    return token_out

# forward process of RL when training transformer 
def _rl_fwd(net, voxels):
    codes = net.encode(voxels)

    outputs = []        
    samples = []

    seq = torch.zeros(voxels.shape[0], 1, device=net.device).long()
    seq[:,0] = net.ex.T2I[net.ex.START_TOKEN]
    
    for ti in range(0, net.ms - 1):

        preds = _rl_infer_prog(net, codes, seq)[:, ti]

        output = torch.log_softmax(preds, dim = 1)
            
        outputs.append(output)

        with torch.no_grad():
            output_probs = torch.softmax(preds, dim = 1)

            if np.random.rand() < net.epsilon:
                # training
                sample = torch.multinomial(output_probs, 1).flatten()
            else:
                # testing
                sample = torch.max(output_probs, 1)[1].flatten()

            sample = sample.detach()
            samples.append(sample)

            nseq = torch.zeros(voxels.shape[0], ti+2).long()
            nseq[:, :ti+1] = seq[:, :ti+1].detach().cpu()
            nseq[:, ti+1] = sample.data.detach().cpu()
            nseq = nseq.detach()
            seq = nseq.to(net.device)                            
                
    return [outputs, samples]
