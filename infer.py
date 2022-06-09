import torch
import numpy as np
from tqdm import tqdm
import utils
import os
import math
from utils import device

# Update the best program data structures by
# Running the inference network over the data

# Best prog structure:
# infer_metric.pt -> metric value of best prog entries
# infer_labels.pt -> tokens of best prog entries
# infer_label_weights -> token masks of best prog entries
# infer_vinput -> visual input of best_prog_entries

def infer_programs(net, data, args, domain):

    path = args.infer_path
    
    net.eval()
    
    pred_labels = torch.zeros((args.train_size, args.seq_len))
    pred_label_weights = torch.zeros((args.train_size, args.seq_len))
    pred_vinput = torch.zeros(tuple([args.train_size] + domain.get_input_shape()))
    pred_metric = torch.zeros(args.train_size)
                    
    results = {}

    # Run inference on all sets, only save predictions for train
    # Val set used for early stopping
    # Test set used for metric reporting
    ITER_DATA = [
        (data.train_eval_iter, True, args.train_size, 'train'),
        (data.val_eval_iter, False, args.eval_size, 'val'),
        (data.test_eval_iter, False, args.eval_size, 'test')
    ]            
    
    for gen, do_write, num, name in ITER_DATA:
        metric = []
        
        utils.log_print(f"Inferring for {name}", args)

        bi = 0

        for vinput in tqdm(gen(), total = math.ceil(num / args.eval_batch_size)):
            # Inference network runs beam search on each entry in vinput, and returns the beam with highest metric against the entry
            pred_tokens, pred_vinp, prd_metric = net.eval_infer_progs(
                vinput, args.infer_beams
            )

            metric += prd_metric
                        
            if do_write:
                for i in range(len(pred_tokens)):
                    pred_labels[bi, :pred_tokens[i].shape[0]] = pred_tokens[i].cpu()
                    pred_label_weights[bi, :pred_tokens[i].shape[0]] = 1.0
                    pred_vinput[bi] = pred_vinp[i].cpu()
                    pred_metric[bi] = prd_metric[i]
                    bi += 1
                    
        a_METRIC = torch.tensor(metric).mean().item()
        utils.log_print(f"{name} {domain.metric_name}: {a_METRIC}", args)
        results[name] = a_METRIC

    # Best Prog data structure reset each epoch
    if args.keep_mode == 'ep':

        torch.save(pred_labels, path + "infer_labels.pt")
        torch.save(pred_label_weights, path + "infer_label_weights.pt")
        torch.save(pred_vinput, path + "infer_vinput.pt")            

    # Update Best Prog data structure depending on metric value
    elif args.keep_mode == 'at':
        fls = os.listdir(path)

        # No past predictions
        if 'infer_labels.pt' not in fls:
            torch.save(pred_metric, path + "infer_metric.pt")
            torch.save(pred_labels, path + "infer_labels.pt")
            torch.save(pred_label_weights, path + "infer_label_weights.pt")
            torch.save(pred_vinput, path + "infer_vinput.pt")    

        else:

            utils.log_print("Saving Predictions", args)
            
            prev_pred_metric = torch.load(path + "infer_metric.pt")
            prev_pred_labels = torch.load(path + "infer_labels.pt")
            prev_pred_label_weights = torch.load(path + "infer_label_weights.pt")
            prev_pred_vinput = torch.load(path + "infer_vinput.pt")

            save_pred_metric = torch.zeros(pred_metric.shape)
            save_pred_labels = torch.zeros(pred_labels.shape)
            save_pred_label_weights = torch.zeros(pred_label_weights.shape)
            save_pred_vinput = torch.zeros(pred_vinput.shape)

            # Decide whether to update each best prog entry dependant on metric comparison
            
            for si in tqdm(range(save_pred_metric.shape[0])):
                
                if domain.comp_metric(prev_pred_metric[si], pred_metric[si]):

                    save_pred_metric[si] = prev_pred_metric[si]
                    save_pred_labels[si] = prev_pred_labels[si]
                    save_pred_label_weights[si] = prev_pred_label_weights[si]
                    save_pred_vinput[si] = prev_pred_vinput[si]

                else:

                    save_pred_metric[si] = pred_metric[si]
                    save_pred_labels[si] = pred_labels[si]
                    save_pred_label_weights[si] = pred_label_weights[si]
                    save_pred_vinput[si] = pred_vinput[si]
                        
            torch.save(save_pred_metric, path + "infer_metric.pt")
            torch.save(save_pred_labels, path + "infer_labels.pt")
            torch.save(save_pred_label_weights, path + "infer_label_weights.pt")
            torch.save(save_pred_vinput, path + "infer_vinput.pt")    
                        
    else:
        assert False, f'bad keep mode {args.keep_mode}'

    return results
