import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import argparse
import sys

device = torch.device('cuda')

# helper function parse arguments
def getArgs(arg_list):       

    parser = argparse.ArgumentParser()
            
    seen = set()
    
    for s,l,d,t in arg_list:        
        parser.add_argument(s, l, default=d, type = t)
        seen.add(s)
        seen.add(l)
            
    args, _ = parser.parse_known_args()
    
    return args

# write a simple point cloud 
def writeSPC(pc, fn):
    with open(fn, 'w') as f:
        for a, b, c in pc:
            f.write(f'v {a} {b} {c} \n')
            

# initialize plad experiment
def init_exp_model_run(args):
    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)

    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')    
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots > /dev/null 2>&1')

    os.system(f'mkdir {args.ws_save_path} > /dev/null 2>&1')
    os.system(f'mkdir {args.infer_path} > /dev/null 2>&1')
    
    with open(f"{args.outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f'CMD: {" ".join(sys.argv)}\n')        
        f.write(f"ARGS: {args}\n")                    

# initialize pretraining experiment
def init_pretrain_run(args):
        
    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)

    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')
    
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')

    with open(f"{args.outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f'CMD: {" ".join(sys.argv)}\n')        
        f.write(f"ARGS: {args}\n")            
        
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/train > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/cad > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/models > /dev/null 2>&1')    
    

def log_print(s, args, fn='log'):
    of = f"{args.outpath}/{args.exp_name}/{fn}.txt"
    with open(of, 'a') as f:
        f.write(f"{s}\n")
    print(s)
    
def print_results(
    LOG_INFO,
    result,
    args
):
    res = ""
    for info in LOG_INFO:
        if len(info) == 3:
            name, key, norm_key = info
            if key not in result:
                continue
            _res = result[key] / (result[norm_key]+1e-8)
                
        elif len(info) == 5:
            name, key1, norm_key1, key2, norm_key2 = info
            if key1 not in result or key2 not in result:
                continue
            res1 = result[key1] / (result[norm_key1]+1e-8)
            res2 = result[key2] / (result[norm_key2]+1e-8)
            _res = (res1 + res2) / 2
                
        else:
            assert False, f'bad log info {info}'
                                     
        res += f"    {name} : {round(_res, 4)}\n"

    log_print(res, args)

    
def make_simp_plots(res, epochs, args, name):
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

    
def make_plots(
    LOG_INFO,
    results,
    plots,
    epochs,
    args,
    fname
):
    for info in LOG_INFO:
        
        for rname, result in results.items():
            if len(info) == 3:
                name, key, norm_key = info
                if key not in result:
                    continue
                res = result[key] / (result[norm_key]+1e-8)
                
            elif len(info) == 5:
                name, key1, norm_key1, key2, norm_key2 = info
                if key1 not in result or key2 not in result:
                    continue
                res1 = result[key1] / (result[norm_key1]+1e-8)
                res2 = result[key2] / (result[norm_key2]+1e-8)
                res = (res1 + res2) / 2
                
            else:
                assert False, f'bad log info {info}'
                        
            if name not in plots[rname]:
                plots[rname][name] = [res]
            else:
                plots[rname][name].append(res)



        plt.clf()
        for key in plots:
            if name not in plots[key]:
                continue
            plt.plot(
                epochs,
                plots[key][name],
                label= key
            )
        plt.legend()
        plt.grid()
        plt.savefig(f'{args.outpath}/{args.exp_name}/plots/{fname}/{name}.png')


