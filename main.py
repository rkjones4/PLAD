import os, sys
import torch
import utils
import json
import pretrain_sl as psl

# Each domain should be imported and returned, depending on the domain_name aegument
def load_domain(args):    

    if args.domain_name == 'csg3d':
        from domains.csg3d import CSG3D_DOMAIN
        return CSG3D_DOMAIN()

    elif args.domain_name == 'sa':
        from domains.shapeAssembly import SA_DOMAIN
        return SA_DOMAIN()

    elif args.domain_name == 'csg2d':
        from domains.csg2d import CSG2D_DOMAIN
        return CSG2D_DOMAIN()
    
    else:
        assert False, f'bad domain name {args.domain_name}'
    
def main():
    main_args = utils.getArgs([
        ('-mm', '--main_mode', None, str), # Set the main mode ['fine_tune', 'pre_train']
        ('-dn', '--domain_name', None, str), # Set the domain ['csg3d', 'sa', 'csg2d']
    ])
    
    domain = load_domain(main_args)

    if main_args.main_mode == 'fine_tune':

        # CSG2D logic is special cased for fair-comparison against CSGNet
        # https://github.com/Hippogriff/CSGNet
        
        if main_args.domain_name == 'csg2d':
            return domain.legacy_fine_tune()
        
        return fine_tune(domain)

    elif main_args.main_mode == 'pretrain':

        assert main_args.domain_name != 'csg2d', 'Pretraining not supported for 2DCSG'
        
        return psl.pretrain_SL(domain)
        
    else:
        assert False, f'bad main main {main_args.main_mode}'

# Fine-tune a recognition network towards a domain of interest
def fine_tune(domain):

    # Load args, rec net, target distribution of real_data
    args = domain.get_ft_args()   
    net = domain.load_pretrain_net()
    real_data = domain.load_real_data()

    # If doing RL, initialize it
    if 'RL' in args.ft_mode:
        domain.init_rl_run() 
        assert args.ft_mode == 'RL'

    # If doing WS, create a generative model
    if 'WS' in args.ft_mode:
        has_gen_model = True
        gen_model = domain.create_gen_model()
    else:
        has_gen_model = False
        gen_model = None
        
    res = {
        'train': [],
        'val': [],
        'test': [],        
    }

    inf_epochs = 0
    gen_epochs = 0
    
    Linf_epochs = []
    Lgen_epochs = []
        
    Round = 0

    best_val = 0.
    best_epoch = 0
            
    os.system(f'rm -f {args.infer_path}/*.pt')
    os.system(f'rm -f {args.ws_save_path}/*.pt')
    
    while inf_epochs < args.max_iters:
        utils.log_print(f"ROUND {Round} (Inf Epochs: {inf_epochs})", args)

        # Run Inf Net over real_data to update best_prog data structure
        with torch.no_grad():
            iter_res = domain.infer_programs(net, real_data)

        # Plotting / eval metric logic
        for part, metric in iter_res.items():
            res[part].append(metric)

        Linf_epochs.append(inf_epochs)

        eres = {k:v for k,v in res.items()}
        eres['epochs'] = Linf_epochs
        
        if has_gen_model:
            Lgen_epochs.append(gen_epochs)
            eres['gen_epochs'] = Lgen_epochs
        
        json.dump(eres, open(f"model_output/{args.exp_name}/res.json" ,'w'))
        del eres
        
        utils.make_simp_plots(res, Linf_epochs, args, domain.metric_name)        
        # End plotting / eval metric logic

        # Update best network version depending on val metric
        if domain.should_save(iter_res['val'], best_val, args.threshold):                            
            utils.log_print("Replacing best model", args)
            best_val = iter_res['val']
            best_epoch = inf_epochs                    
            torch.save(net.state_dict(), f"model_output/{args.exp_name}/inf_net.pt")

        # Stop early baes on val metric
        if inf_epochs - best_epoch > args.iter_patience:
            utils.log_print("Stopping early", args)
            break                    
        
        if args.ft_mode == 'RL':
            # Fine-tuning RL
            inf_epochs += domain.train_rl(net, real_data)

        else:
            # Fine-tuning PLAD
            _inf_epochs, _gen_epochs = domain.train_plad(
                net,
                gen_model,
                real_data,
                has_gen_model,
            )
        
            inf_epochs += _inf_epochs
            gen_epochs += _gen_epochs
            
        Round += 1

    os.system(f'rm -f {args.infer_path}/*.pt')
            
if __name__ == '__main__':    
    main()

