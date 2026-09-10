import os
from timeit import default_timer as timer

import hydra
import torch
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.utilities import rank_zero_only

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
# from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from utils.helpers import print_status, count_model_parameters, ensure_path_join, denormalize_to_zero_to_one, \
    normalize_to_neg_one_to_one

import torchmetrics as tm

from torchvision.transforms import Resize, ToTensor 
from PIL import Image 
from natsort import natsorted 
import sys
sys.path.insert(0, 'ID_generator/')

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


class DiffusionTrainerLite(LightningLite):

    @staticmethod
    def restore_checkpoint(model, optimizer, path, lr_scheduler=None):
        model_ckpt = torch.load(os.path.join(path, 'checkpoints', 'model.ckpt'), map_location="cpu")
        optimization_ckpt = torch.load(os.path.join(path, 'checkpoints', 'optimization.ckpt'), map_location="cpu")

        global_step = optimization_ckpt['global_step']
        epoch = optimization_ckpt['epoch']

        new_model_ckpt={}
        for k, v in model_ckpt.items():
            new_k = k.replace('_module.','') if '_module' in k else k
            new_model_ckpt[new_k] = v
        model.load_state_dict(new_model_ckpt) 
        optimizer.load_state_dict(optimization_ckpt['optimizer'])

        if 'lr_scheduler' in optimization_ckpt:
            lr_scheduler.load_state_dict(optimization_ckpt['lr_scheduler'])

        print(f"Successfully restored checkpoint (global_step: {global_step}) from: {path}")

        return global_step, epoch

    @rank_zero_only
    def generate_new_identity(self, model, size, latent_decoder=None, save_path=None, context_size = 0, gener_path=None, gener_context_path=None):
        
        
        model.eval()
        new_size = Resize([1024, 400]) 
        with torch.no_grad():
            
            context = torch.randn(len(save_path), context_size)
            
            syn_context = torch.nn.functional.normalize(torch.randn_like(context)).cuda()
            samples_syn_cond = model.sample(len(save_path), size, context=syn_context).cpu()
            if latent_decoder is not None:
                samples_syn_cond = latent_decoder(samples_syn_cond)
            samples_syn_cond = denormalize_to_zero_to_one(samples_syn_cond)
            samples_syn_cond = new_size(samples_syn_cond) 
            for index in range(samples_syn_cond.shape[0]):
                save_image(samples_syn_cond[index], gener_path+'/'+save_path[index]+f".jpg")
                torch.save(syn_context[index], gener_context_path+'/'+save_path[index]+f".npy")
                

    def run(self, cfg):
        # seed for reproducibility
        self.seed_everything(cfg.constants.seed)

        # create diffusion model from config
        diffusion_model = instantiate(cfg.diffusion)

        # count number of parameters
        trainable_params, _, total_params = count_model_parameters(diffusion_model)
        print(f"#Params Diffusion Model: {trainable_params} (Total: {total_params})")

        # create optimizer from config
        partial_optimizer = instantiate(cfg.training.optimizer)
        optimizer = partial_optimizer(params=diffusion_model.parameters())

        if cfg.training.lr_scheduler is not None:
            partial_lr_scheduler = instantiate(cfg.training.lr_scheduler)
            lr_scheduler = partial_lr_scheduler(optimizer=optimizer)
        else:
            lr_scheduler = None

        # registrate model and optimizer in lite
        diffusion_model, optimizer = self.setup(diffusion_model, optimizer)

        # (optional) load pre-existing weights
        if cfg.training.checkpoint.restore:
            global_step, epoch = self.restore_checkpoint(diffusion_model.module, optimizer, cfg.training.checkpoint.path, lr_scheduler)
        else:
            global_step, epoch = 0, 0

        # create exponential moving average (ema) model
        partial_ema = instantiate(cfg.training.ema)
        ema_model = partial_ema(model=diffusion_model.module)
        ema_model.optimization_step += global_step

        if cfg.training.checkpoint.restore:
            ema_model_ckpt = torch.load(os.path.join(cfg.training.checkpoint.path, 'checkpoints', 'ema_averaged_model.ckpt'), map_location="cpu")
            ema_model.averaged_model.load_state_dict(ema_model_ckpt)

        if cfg.latent_diffusion:
            # create VQGAN encoder and decoder for training in its latent space
            latent_encoder = VQEncoderInterface(
                first_stage_config_path=os.path.join(cfg.paths.root, "models", "autoencoder", "first_stage_config.yaml"),
                encoder_state_dict_path=os.path.join(cfg.paths.root, "models", "autoencoder", "first_stage_encoder_state_dict.pt")
            )
            latent_decoder = VQDecoderInterface(
                first_stage_config_path=os.path.join(cfg.paths.root, "models", "autoencoder", "first_stage_config.yaml"),
                decoder_state_dict_path=os.path.join(cfg.paths.root, "models", "autoencoder", "first_stage_decoder_state_dict.pt")
            )

            # only push encoder to GPU to save memory (decoder is only used for sampling and not during training)
            self.setup(latent_encoder)

            # set both into evaluation mode
            latent_encoder.eval()
            latent_decoder.eval()

            # display their number of parameters
            trainable_params, _, total_params = count_model_parameters(latent_encoder)
            print(f"#Params Latent Encoder Model: {trainable_params} (Total: {total_params})")
            trainable_params, _, total_params = count_model_parameters(latent_decoder)
            print(f"#Params Latent Decoder Model: {trainable_params} (Total: {total_params})")
        else:
            latent_encoder, latent_decoder = None, None

       
        batch_size = 24
        
        while True:
            
            for itea_index in range(10):  # itea_num
                print(itea_index)
                batch_npys = [str(itea_index*batch_size + i) for i in range(batch_size)]
                
                
                if cfg.latent_diffusion:
                    
                    
                    with torch.no_grad():
                        sample_size = torch.empty(3,32,32).shape 
                else:
                    sample_size = torch.empty(3,128,128).shape 
                
                
                self.generate_new_identity(ema_model.averaged_model, size=sample_size,
                                                    latent_decoder=latent_decoder,
                                                    save_path=batch_npys, context_size=cfg.model.context_input_channels, 
                                                    gener_path=cfg.gener_path, gener_context_path=cfg.gener_context_path)  
                
            break


@hydra.main(config_path='configs', config_name='person_config', version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    trainer = DiffusionTrainerLite(devices='auto', accelerator='gpu', precision=cfg.training.precision)
    trainer.run(cfg)


if __name__ == "__main__":

    train()
