import torch
import numpy as np
from leg_src.Models.models import ImitateJoint, ParseModelOutput
from leg_src.utils.train_utils import prepare_input_op, chamfer, beams_parser, image_from_expressions
from leg_src.utils.generators.shapenet_generater import Generator
from tqdm import tqdm
import utils
import os

from utils import device

"""
Infer programs on cad dataset
"""
def infer_programs(imitate_net, args, sample_fn='beam', only_train=False):

    path = args.infer_path
    
    max_len = args.max_len
    beam_width = args.beam_width

    # Load the terminals symbols of the grammar
    with open("legacy/terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]
    
    imitate_net.eval()
    imitate_net.epsilon = 0
    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              args.canvas_shape)
    
    pred_labels = np.zeros((args.train_size, args.num_inst_progs, max_len))
    pred_images = np.zeros((args.train_size, args.num_inst_progs, 64, 64))
    pred_cds = np.zeros((args.train_size, args.num_inst_progs))
    
    generator = Generator()

    train_gen = generator.train_gen(
        batch_size=args.batch_size,
        path=args.data_path)

    val_gen = generator.val_gen(
        batch_size=args.batch_size,
        path=args.data_path)
    
    test_gen = generator.test_gen(
        batch_size=args.batch_size,
        path=args.data_path)
            
    Target_images = []
    

    results = {}

    if only_train:
        ITER_DATA = [
            (train_gen, True, args.train_size, 'train'),
        ]
    else:
        ITER_DATA = [
            (train_gen, True, args.train_size, 'train'),
            (val_gen, False, args.test_size, 'val'),
            (test_gen, False, args.test_size, 'test')
        ]            
    
    for gen, do_write, num, name in ITER_DATA:
        CDs = 0
        
        utils.log_print(f"Inferring for {name}", args)
        
        for batch_idx in tqdm(range(num // args.batch_size)):
            with torch.no_grad():
                
                data_ = next(gen)
                labels = np.zeros((args.batch_size, max_len), dtype=np.int32)
                one_hot_labels = prepare_input_op(labels, len(unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = torch.from_numpy(data_).to(device)

                if sample_fn == 'beam':
                    prog_samp_fn = imitate_net.beam_search
                    
                elif sample_fn == 'mc':
                    prog_samp_fn = imitate_net.monte_carlo
                                
                all_beams, _, _ = prog_samp_fn(
                    [data[-1, :, 0, :, :], one_hot_labels],
                    beam_width,
                    max_len
                )

                beam_labels = beams_parser(
                    all_beams, data_.shape[1], beam_width=beam_width)

                beam_labels_numpy = np.zeros(
                    (args.batch_size * beam_width, max_len), dtype=np.int32)

                if do_write:
                    Target_images.append(data_[-1, :, 0, :, :])
                
                for i in range(data_.shape[1]):
                    beam_labels_numpy[i * beam_width:(
                        i + 1) * beam_width, :] = beam_labels[i]

                # find expression from these predicted beam labels
                expressions = [""] * args.batch_size * beam_width
                for i in range(args.batch_size * beam_width):
                    for j in range(max_len):
                        expressions[i] += unique_draw[beam_labels_numpy[i, j]]
                for index, prog in enumerate(expressions):
                    expressions[index] = prog.split("$")[0]

                predicted_images = image_from_expressions(parser, expressions)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
                target_images_new = np.repeat(
                    target_images, axis=0, repeats=beam_width)

                beam_CD = chamfer(target_images_new, predicted_images)
                
                if do_write:
                    # select best expression by chamfer distance
                    best_labels = np.zeros((args.batch_size, max_len))
                    for r in range(args.batch_size):
                        inst_CD = beam_CD[r * beam_width:(r + 1) * beam_width]

                        for iii, idx in enumerate(np.argsort(inst_CD)[:args.num_inst_progs]):  
                            idx = np.argmin(inst_CD)
                            best_labels[r] = beam_labels[r][idx]
                            pred_labels[batch_idx*args.batch_size:batch_idx*args.batch_size + args.batch_size, iii] = best_labels

            CD = np.zeros((args.batch_size, 1))
            for r in range(args.batch_size):
                CD[r, 0] = min(beam_CD[r * beam_width:(r + 1) * beam_width])
                if do_write:
                    inst_CD = beam_CD[r*beam_width:(r+1)*beam_width]
                    for iii, idx in enumerate(np.argsort(inst_CD)[:args.num_inst_progs]):
                        pred_cds[batch_idx*args.batch_size+r, iii] = inst_CD[idx]
                        pred_images[batch_idx*args.batch_size+r, iii] = predicted_images[
                            r*beam_width + idx
                        ]

            CDs += np.mean(CD)

        a_CD = CDs / (num // args.batch_size)
        utils.log_print(f"{name} CD: {a_CD}", args)
        results[name] = a_CD

    target_images = np.concatenate(Target_images, axis=0)
    torch.save(target_images, path + "target_images.pt")
    
    if args.keep_mode == 'ep':

        torch.save(pred_labels, path + "infer_labels.pt")
        torch.save(pred_images, path + "infer_images.pt")            

    elif args.keep_mode == 'at':
        fls = os.listdir(path)

        if 'infer_labels.pt' not in fls:
            torch.save(pred_cds, path + "infer_cds.pt")
            torch.save(pred_labels, path + "infer_labels.pt")
            torch.save(pred_images, path + "infer_images.pt")    

        else:

            utils.log_print("Saving Predictions", args)
            
            prev_pred_cds = torch.load(path + "infer_cds.pt")
            prev_pred_labels = torch.load(path + "infer_labels.pt")
            prev_pred_images = torch.load(path + "infer_images.pt")

            save_pred_cds = np.zeros(pred_cds.shape)
            save_pred_labels = np.zeros(pred_labels.shape)
            save_pred_images = np.zeros(pred_images.shape)
            
            for si in tqdm(range(save_pred_cds.shape[0])):

                ai = 0
                bi = 0

                for pi in range(save_pred_cds.shape[1]):
                    if prev_pred_cds[si, ai] <= pred_cds[si, bi]:
                        save_pred_cds[si, pi] = prev_pred_cds[si, ai]
                        save_pred_labels[si, pi] = prev_pred_labels[si, ai]
                        save_pred_images[si, pi] = prev_pred_images[si, ai]
                        ai += 1
                    else:
                        save_pred_cds[si, pi] = pred_cds[si, bi]
                        save_pred_labels[si, pi] = pred_labels[si, bi]
                        save_pred_images[si, pi] = pred_images[si, bi]
                        bi += 1
                        
            torch.save(save_pred_cds, path + "infer_cds.pt")
            torch.save(save_pred_labels, path + "infer_labels.pt")
            torch.save(save_pred_images, path + "infer_images.pt")    
                        
    else:
        assert False
                        
    return results
