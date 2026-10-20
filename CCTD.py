import torch
import numpy as np
from torch import optim
from time import time
from utils import loadmat, evaluation, set_all_seeds, min_max_normalize
from utils import add_gaussian_noise, add_non_iid_gaussian_noise
from utils import add_impulse_noise, add_stripe_noise, add_deadline_noise
from network import CTN
import scipy.io as sio
from pathlib import Path
import json


def _json_default(value):
    """Convert NumPy values produced by metric functions to JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_and_preprocess_data(file_path, noise_case, hsi_key='hsi', gde_key='msi'):
    """Load and preprocess input data"""
    data = loadmat(file_path)
    
    # Extract and normalize images
    I_GT = data[hsi_key].astype(np.float32)
    I_Guide = data[gde_key].astype(np.float32)
    
    # Ensure guide has channel dimension
    if I_Guide.ndim == 2:
        I_Guide = I_Guide[..., None]
    
    from skimage.transform import resize
    I_Guide = resize(I_Guide, I_GT.shape[:2])
    # print(I_Guide.shape)
    
    # Normalize to [0, 1] range
    I_GT = min_max_normalize(I_GT)
    I_Guide = min_max_normalize(I_Guide)
    
    # Add noise based on case
    if noise_case == 1:
        N_HSI = add_gaussian_noise(I_GT, 10/255)
    elif noise_case == 2:
        N_HSI = add_non_iid_gaussian_noise(I_GT, (5/255,50/255), 2/3)
        N_HSI = add_impulse_noise(N_HSI,  0.2, 1/3)
        N_HSI = add_deadline_noise(N_HSI, 0.2, 1/3)
        N_HSI = add_stripe_noise(N_HSI,   0.2, 1/3)
    elif noise_case == 3:
        N_HSI = add_non_iid_gaussian_noise(I_GT, (5/255,50/255), 2/3)
        N_HSI = add_impulse_noise(N_HSI,  0.3, 1/3)
        N_HSI = add_deadline_noise(N_HSI, 0.3, 1/3)
        N_HSI = add_stripe_noise(N_HSI,   0.3, 1/3)
    elif noise_case == 4:
        N_HSI = add_non_iid_gaussian_noise(I_GT, (5/255,50/255), 2/3)
        N_HSI = add_impulse_noise(N_HSI,  0.4, 1/3)
        N_HSI = add_deadline_noise(N_HSI, 0.4, 1/3)
        N_HSI = add_stripe_noise(N_HSI,   0.4, 1/3)
    elif noise_case == 5:
        N_HSI = add_non_iid_gaussian_noise(I_GT, (5/255,50/255), 2/3)
        N_HSI = add_impulse_noise(N_HSI,  0.5, 1/3)
        N_HSI = add_deadline_noise(N_HSI, 0.5, 1/3)
        N_HSI = add_stripe_noise(N_HSI,   0.5, 1/3)
    elif noise_case == 6:
        N_HSI = add_non_iid_gaussian_noise(I_GT, (5/255,50/255), 2/3)
        N_HSI = add_impulse_noise(N_HSI,  (0.05,0.5), 1/3)
        N_HSI = add_deadline_noise(N_HSI, (0.05,0.5), 1/3)
        N_HSI = add_stripe_noise(N_HSI,   (0.05,0.5), 1/3)
    
    N_HSI = N_HSI.copy()
    return I_GT, I_Guide, N_HSI
    
class HSIDenoisingPipeline:
    def __init__(self, config):
        # Configuration
        config = self.check_config(config)
        self.config = config

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Initialize components
        self.net = None
        self.recon_loss = self._get_reon_loss_fn()
        self.optimizer = None
        self.output_dir = None  # Will be set during pipeline execution
        self.psnr_history = []  # To track PSNR over iterations
    
    def check_config(self, config):
        # default_loss_weight = {1: {'weight_recon': 1,'weight_guide': 0.130,'weight_nn': 0.008},
        #                        2: {'weight_recon': 1,'weight_guide': 120,'weight_nn': 80},
        #                        3: {'weight_recon': 1,'weight_guide': 110,'weight_nn': 100},
        #                        4: {'weight_recon': 1,'weight_guide': 100,'weight_nn': 120},
        #                        5: {'weight_recon': 1,'weight_guide': 90, 'weight_nn': 130},
        #                        6: {'weight_recon': 1,'weight_guide': 110, 'weight_nn':100},
        #                        }
        default_loss_weight = {1: {'weight_recon': 1,'weight_guide': 0.1,'weight_nn': 0.04},
                       2: {'weight_recon': 1,'weight_guide': 200,'weight_nn': 90},
                       3: {'weight_recon': 1,'weight_guide': 200,'weight_nn': 90},
                       4: {'weight_recon': 1,'weight_guide': 200,'weight_nn': 90},
                       5: {'weight_recon': 1,'weight_guide': 200, 'weight_nn': 90},
                       6: {'weight_recon': 1,'weight_guide': 200, 'weight_nn':90},
                       }
        noise_case = config['noise_case']
        if 'weight_recon' not in config:
            config['weight_recon'] = default_loss_weight[noise_case]['weight_recon']
        if 'weight_guide' not in config:
            config['weight_guide'] = default_loss_weight[noise_case]['weight_guide']
        if 'weight_nn' not in config:
            config['weight_nn'] = default_loss_weight[noise_case]['weight_nn']
        return config
        
    def _get_reon_loss_fn(self):
        return 'l2' if self.config['noise_case'] in [1] else 'l1'
    
    def _create_output_directory(self):
        """Create output directory structure based on data path and noise case"""
        data_name = self.config['data_name']
        output_dir = Path(self.config['output_root']) / data_name / f'case{self.config["noise_case"]}'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
    
    def _save_preprocessed_data(self, I_HSI, I_GDE, N_HSI, save_path=None):
        """Save preprocessed data to mat file"""
        if save_path is None:
            save_path = self.output_dir / 'data.mat'
        sio.savemat(save_path, {
            'I_HSI': I_HSI,
            'I_GDE': I_GDE,
            'N_HSI': N_HSI,
            'noise_case': self.config['noise_case']
        })
        print(f"Saved preprocessed data to {save_path}")
        
    def _save_metrics(self, metrics, save_path=None):
        if save_path is None:
            save_path = self.output_dir / 'metrics.json'
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=_json_default)
        print(f"Saved metrics to {save_path}")
        
    def _save_psnr_history(self, save_path=None):
        if save_path is None:
            save_path = self.output_dir / 'psnr_history.json'
        with open(save_path, 'w') as f:
            json.dump(self.psnr_history, f, indent=4, default=_json_default)
        print(f"Saved PSNR history to {save_path}")

    def _save_results(self, denoised_img, save_path=None):
        """Save denoised image and metrics"""
        # Save denoised image
        if save_path is None:
            save_path = self.output_dir / f"CCTD_{self.config['gde_key']}.mat"
        sio.savemat(save_path, {'denoised': denoised_img})
        
        print(f"Saved results to {save_path}")

    def setup_network(self, hsi_shape, guide_shape):
        """Initialize network and related components"""
        tv_shape = guide_shape[:2]
        self.net = CTN(
            hsi_shape, guide_shape, tv_shape, hsi_shape[-1], omega=self.config['omega'], device=self.device
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config['lr'])

    
    def _compute_loss(self, out_HSI, out_Guide, X, Guide):
        """Compute all loss components"""
        mask = X != 0
        
        # Reconstruction loss
        if self.recon_loss == 'l2':
            recon_loss = torch.norm(out_HSI[mask] - X[mask], 2)
        else:  # l1
            recon_loss = torch.norm(out_HSI[mask] - X[mask], 1)
        
        # Guide and nuclear norm losses
        guide_loss =  torch.norm(out_Guide - Guide, 2)
        # grad_x, grad_y = grad_fn(out_Guide - Guide)
        # guide_loss =  torch.norm(grad_x, 2) + torch.norm(grad_y, 2)
        
        nn_loss =  torch.norm(out_HSI.reshape(-1, out_HSI.shape[-1]), 'nuc')
        
        # tv loss
        # grad_x, grad_y = grad_fn(HSI4TV) 
        # grad_gx,grad_gy= grad_fn(GDE4TV.mean(-1,True))
        # W1 = (1-grad_gx.abs()).pow(5)
        # W2 = (1-grad_gy.abs()).pow(5)
        # tv_loss = (W1*grad_x).abs().sum() + (W2*grad_y).abs().sum()
        
        # total loss
        # total_loss = self.config['weight_recon'] * recon_loss + \
        #              0 * guide_loss + \
        #              self.config['weight_nn']    * nn_loss 
                     
        total_loss = self.config['weight_recon'] * recon_loss + \
                      self.config['weight_guide'] * guide_loss + \
                      self.config['weight_nn']    * nn_loss 
        return total_loss

    def train(self, X, Guide, I_GT):
        """Main training loop"""
        self.psnr_history = []
        t0 = time()
        last_10_percent = int(0.9 * self.config['max_iter'])
        output_sum, count = None, 0
        
        for iter in range(self.config['max_iter']):
            # Forward pass
            out_HSI, out_Guide = self.net()
            # out_HSI, out_Guide, HSI4TV, GDE4TV = self.net()
            
            # Accumulate last 10% iterations
            if iter >= last_10_percent:
                if iter == last_10_percent:
                    output_sum = out_HSI.detach().cpu().numpy()
                    count = 1
                else:
                    output_sum += out_HSI.detach().cpu().numpy()
                    count += 1

            # Loss computation and backpropagation
            loss = self._compute_loss(out_HSI, out_Guide, X, Guide)
            # loss = self._compute_loss(out_HSI, out_Guide, X, Guide, HSI4TV, GDE4TV)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            torch.cuda.empty_cache()
            
            # Evaluation and logging
            if iter % self.config['eval_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = evaluation(
                    I_GT,
                    out_HSI.detach().cpu().numpy(),
                    ['psnr', 'ssim', 'ergas', 'sam'],
                    verbose_prefix=f'iter: {iter:04d}, lr: {current_lr:.2e}, time: {int(time()-t0)}s, '
                )
                self.psnr_history.append({
                    'iteration': iter,
                    'psnr': metrics['psnr'],
                    'time': time() - t0
                })
        
        # Final metrics calculation
        final_out = output_sum / count
        final_metrics = evaluation(
            I_GT,
            final_out,
            ['psnr', 'ssim', 'ergas', 'sam'],
            verbose_prefix='Final results: '
        )
        final_metrics['time'] = time() - t0
        
        self.final_out = final_out
        self.final_metrics = final_metrics
    
    def run_pipeline(self, I_GT, I_Guide, N_HSI):
        """Complete denoising pipeline"""
        # Create output directory
        self._create_output_directory()
        
        # save data
        self._save_preprocessed_data(I_GT, I_Guide, N_HSI)
        
        # Convert to tensors and setup network
        X = torch.tensor(N_HSI).to(self.device)
        Guide = torch.tensor(I_Guide).to(self.device)
        self.setup_network(X.shape, Guide.shape)
        
        # Train and save results
        self.train(X, Guide, I_GT)
        self._save_results(self.final_out)
        self._save_metrics(self.final_metrics)
        self._save_psnr_history()


if __name__ == "__main__":
    ##
    # config = {
    #     # data
    #     'data_name': 'Beijing',
    #     'gde_key': 'msi',
    #     'hsi_key': 'hsi',
    #     # noise simulation
    #     'noise_case': 6,
    #     # network parameter
    #     'omega': 2,
    #     # train 
    #     'lr': 1e-4,
    #     'max_iter': 5001,
    #     # loss 
    #     'weight_recon': 1,
    #     'weight_guide': 110,
    #     'weight_nn': 90,
    #     # 'weight_tv': 6e-5,
    #     # others
    #     'seed': 3407,
    #     'eval_interval': 100,
    #     'output_root': 'output', 
    # }
    # set_all_seeds(config['seed'])
    # file_path = f'data/{config["data_name"]}.mat'
    # I_GT, I_Guide, N_HSI = load_and_preprocess_data(file_path,
    #                                                 config['noise_case'],
    #                                                 config['hsi_key'],
    #                                                 config['gde_key'])
    # pipeline = HSIDenoisingPipeline(config)
    # pipeline.run_pipeline(I_GT, I_Guide, N_HSI)
    
    ##
    for data_name in ['Beijing', 'Wuhan', 'Florence']:  #  'August'
        if data_name in ['Beijing', 'Wuhan']:
            gde_list = ['msi', 'sar']
        elif data_name in ['Florence']:
            gde_list = ['pan']
        elif data_name in ['August']:
            gde_list = ['lidar']
            
        for gde_key in gde_list:
                for noise_case in range(1,7):
                    config = {
                        # data
                        'data_name': data_name,
                        'gde_key': gde_key,
                        'hsi_key': 'hsi',
                        # noise simulation
                        'noise_case': noise_case,
                        # network parameter
                        'omega': 3,
                        # train 
                        'lr': 1e-4,
                        'max_iter': 5001,
                        # loss 
                        # case 1
                        # 'weight_recon': 1,
                        # 'weight_guide': 0.13,
                        # 'weight_nn': 0.008,
                        # case 2
                        # 'weight_recon': 1,
                        # 'weight_guide': 0.13,
                        # 'weight_nn': 0.06,
                        # case 3
                        # 'weight_recon': 1,
                        # 'weight_guide': 0.076,
                        # 'weight_nn': 0.08,
                        # case 4
                        # 'weight_recon': 1,
                        # 'weight_guide': 130,
                        # 'weight_nn': 60,
                        # case 5
                        # 'weight_recon': 1,
                        # 'weight_guide': 120,
                        # 'weight_nn': 80,
                        # case 6
                        # 'weight_recon': 1,
                        # 'weight_guide': 110,
                        # 'weight_nn': 90,
                        # others
                        'seed': 3407,
                        'eval_interval': 100,
                        'output_root': 'output', 
                    }
                    set_all_seeds(config['seed'])
                    file_path = f'data/{config["data_name"]}.mat'
                    I_GT, I_Guide, N_HSI = load_and_preprocess_data(file_path,
                                                                    config['noise_case'],
                                                                    config['hsi_key'],
                                                                    config['gde_key'])
                    pipeline = HSIDenoisingPipeline(config)
                    pipeline.run_pipeline(I_GT, I_Guide, N_HSI)
                    
