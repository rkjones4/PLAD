import sys, os, torch, json, time, random, ast, utils, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import device
from copy import deepcopy, copy
from tqdm import tqdm

celoss = torch.nn.CrossEntropyLoss(reduction='none')
ex = None


# Code for pretraining the recongition model on synthetically generated data, with supervised learning

def model_eval(
    args,
    loader,
    net,
    e,
    name,
    model_eval_fn,
):

    epoch = e
    
    point_results = []

    corr = 1e-8
    total = 1e-8

    iou = 0.
    
    for count, (voxels, gt_tokens) in enumerate(loader):

        eval_inp = {
            'net': net,
            'voxels': voxels
        }
        
        # preds: tensor where each point is given a terminal 
        pred_tokens = model_eval_fn(eval_inp)        
        
        try:
            prog = ' '.join([ex.I2T[p.item()] for p in pred_tokens])        
            pred_voxels = torch.from_numpy(ex.execute(prog)).to(voxels.device)
            voxels = voxels > 0
            pred_voxels = pred_voxels > 0
            inter = (voxels & pred_voxels).sum().item()
            union = (voxels | pred_voxels).sum().item()
            _iou = inter * 1. / union
                        
        except Exception as e:
            pred_voxels = None
            _iou = 0.

        if gt_tokens is not None:
            pred_tokens = pred_tokens.to(gt_tokens.device)
            gt_prog = ' '.join([ex.I2T[p.item()] for p in gt_tokens])
            mlen = min(gt_tokens.shape[0], pred_tokens.shape[0])        
            _total = max(gt_tokens.shape[0], pred_tokens.shape[0])
            _corr = (gt_tokens[:mlen] == pred_tokens[:mlen]).sum().item()
            corr += _corr
            total += _total
            
        iou += _iou
                                    

    metric_result = {}
    metric_result['iou'] = iou
    metric_result['corr'] = corr
    metric_result['total'] = total
    
    metric_result['count'] = count + 1
    metric_result['nc'] = 1
    
    return metric_result

def run_eval_epoch(
    args,
    res,
    net,
    eval_data,
    EVAL_LOG_INFO,
    e,
    model_eval_fn,
    sub_fn = model_eval
):
        
    if (e+1) % args.eval_per != 0:
        return -1
        
    with torch.no_grad():
        
        net.eval()        
                    
        t = time.time()                

        eval_results = {}
        for key, loader in eval_data:

            eval_results[key] = sub_fn(
                args,
                loader,
                net,
                e,
                key,
                model_eval_fn
            )
            
            utils.log_print(
                f"Evaluation {key} set results:",
                args
            )

            utils.print_results(
                EVAL_LOG_INFO,
                eval_results[key],
                args
            )
                        
        utils.log_print(f"Eval Time = {time.time() - t}", args)

        res['eval_epochs'].append(e)
                
        utils.make_plots(
            EVAL_LOG_INFO,
            eval_results,            
            res['eval_plots'],
            res['eval_epochs'],
            args,
            'eval'
        )
    
    eps = res['eval_epochs']
    metric_res = torch.tensor(res['eval_plots']['val'][args.es_metric])
    cur_ep = eps[-1]
    
    for i, ep in enumerate(eps):
        if cur_ep - ep <= args.es_patience:
            metric_res[i] -= args.es_threshold

    best_ep_ind = metric_res.argmax().item()
    best_ep = eps[best_ep_ind]

    # early stopping logic
    
    if cur_ep - best_ep > args.es_patience:
        utils.log_print(
            f"Stopping early at epoch {cur_ep}, "
            f"choosing epoch {best_ep} with val {args.es_metric} " 
            f"of {metric_res.max().item()}",
            args
        )
        utils.log_print(
            f"Final test val for {args.es_metric} : {res['eval_plots']['val'][args.es_metric][best_ep_ind]}",
            args
        )
        return best_ep

    return -1

def model_train(loader, net, opt, batch_train_fn):
    
    if opt is None:
        net.eval()
    else:
        net.train()
        
    ep_result = {}
    bc = 0.
    for batch in loader:
        bc += 1.

        batch_result = batch_train_fn(batch, net, opt)
        
        for key in batch_result:                        
            if key not in ep_result:                    
                ep_result[key] = batch_result[key]
            else:
                ep_result[key] += batch_result[key]

                
    ep_result['batch_count'] = bc

    return ep_result

def run_train_epoch(
    args, res, net, opt, train_loader, val_loader, TRAIN_LOG_INFO, e, batch_train_fn,
):

    do_print = (e+1) % args.print_per == 0
    
    json.dump(res, open(f"{args.outpath}/{args.exp_name}/res.json" ,'w'))

    t = time.time()
    
    if do_print:
        utils.log_print(f"\nBatch Iter {e}:", args)
            
    train_result = model_train(
        train_loader,
        net,
        opt,
        batch_train_fn
    )

    if do_print:            
        res['train_epochs'].append(e)

        with torch.no_grad():
            val_result = model_train(
                val_loader,
                net,
                None,
                batch_train_fn
            )
        
        ep_result = {
            'train': train_result,
            'val': val_result
        }
            
        utils.log_print(
            f"Train results: ", args
        )
            
        utils.print_results(
            TRAIN_LOG_INFO,
            train_result,
            args,
        )

        utils.log_print(
            f"Val results: ", args
        )
            
        utils.print_results(
            TRAIN_LOG_INFO,
            val_result,
            args,
        )
                 
        utils.make_plots(
            TRAIN_LOG_INFO,
            ep_result,
            res['train_plots'],
            res['train_epochs'],
            args,
            'train'
        )
            
        utils.log_print(
            f"    Time = {time.time() - t}",
            args
        )

            
def model_eval(eval_inp):
    net = eval_inp['net']
    voxels = eval_inp['voxels']
    
    return net.eval_infer_progs(voxels.unsqueeze(0), None)[0][0]

    
def model_train_batch(batch, net, opt):

    voxels, progs, weights = batch

    codes = net.encode(voxels)

    preds = net.infer_prog(codes, progs)

    flat_preds = preds[:,:-1,:].reshape(-1,preds.shape[2])
    flat_targets = progs[:,1:].flatten()    
    flat_weights = weights[:,1:].flatten()


    # MLE updates
    
    loss = (celoss(flat_preds, flat_targets) * flat_weights).sum() / (flat_weights.sum() + 1e-8)
        
    br = {}

    with torch.no_grad():
        br['corr'] = ((flat_preds.argmax(dim=1) == flat_targets).float() * flat_weights).sum().item()
        br['total'] = flat_weights.sum().item()
        br['loss'] = loss.item()
        
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return br
    
def pretrain_SL(domain):
    args = domain.get_pt_args()
    net = domain.load_new_net()
    global ex
    ex = domain.executor    

    # synthetic data sampled from the grammar randomly
    
    train_loader, val_loader = domain.get_synth_datasets(args)        
    
    net.to(device)
    
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        eps = 1e-6
    )
    
    res = {
        'train_plots': {'train':{}, 'val':{}},
        'train_epochs': [],
        'eval_plots': {'train':{}, 'val':{}},
        'eval_epochs': []
    }

    save_model_weights = {}
        
    eval_data = [
        ('train', train_loader),
        ('val', val_loader),
    ]
    
    print("Starting Training")
    
    for e in range(args.num_iters):

        train_loader.mode = 'train'
        val_loader.mode = 'train'
        
        run_train_epoch(
            args,
            res,
            net,
            opt,
            train_loader,
            val_loader,
            domain.TRAIN_LOG_INFO,
            e,
            model_train_batch
        )
        
        train_loader.mode = 'eval'
        val_loader.mode = 'eval'
        
        best_ep = run_eval_epoch(
            args,
            res,
            net,
            eval_data,
            domain.EVAL_LOG_INFO,
            e,
            model_eval
        )
                
        if best_ep >= 0:
            break

        if (e+1) % args.eval_per == 0:
            save_model_weights[e] = deepcopy(net.state_dict())
            torch.save(
                net.state_dict(),
                f"{args.outpath}/{args.exp_name}/models/net_{e}.pt"
            )
            
    utils.log_print("Saving Best Model", args)
            
    if best_ep < 0:
        best_ep = res['eval_epochs'][-1]                
    
    torch.save(
        save_model_weights[best_ep],
        f"{args.outpath}/{args.exp_name}/models/best_net.pt"
    )
            
