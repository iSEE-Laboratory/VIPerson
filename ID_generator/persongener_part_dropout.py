import os
from timeit import default_timer as timer

import hydra
import torch
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.utilities import rank_zero_only

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

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
    def generate_hard_identity(self, model, size, x=None, context=None, latent_decoder=None, save_path=None,  N_PER_ROW=0, gener_hard_path=None):
       
        ourdrop = torch.nn.Dropout(p=0.25)
        
        
        new_size = Resize([1024, 400]) 
        with torch.no_grad():
            
            if context is not None:
                
                context_fixed = context.repeat(N_PER_ROW, 1).cuda()
                context_fixed = ourdrop(context_fixed)
                samples_fixed_context = model.sample(N_PER_ROW * context.shape[0], size, context=context_fixed).cpu() 
                if latent_decoder is not None:
                    samples_fixed_context = latent_decoder(samples_fixed_context)
                samples_fixed_context = denormalize_to_zero_to_one(samples_fixed_context)
                samples_fixed_context = new_size(samples_fixed_context)  
                for index in range(samples_fixed_context.shape[0]):
                    save_image(samples_fixed_context[index], gener_hard_path+save_path[index%len(context)].replace('.npy', '')+f"_{int(index/len(context)):02d}.jpg")
                

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

        
        samples_dir = cfg.gener_path_text_emb
        images_dir = cfg.gener_path
        
        npy_names = natsorted(os.listdir(samples_dir))

        print(len(npy_names))
        # =================================================

        batch_size = 8
        itea_num = int(len(npy_names)/batch_size) + 1
        print(itea_num)
        
        x_visualisation = None
        context_visualisation = None
        
        transf = ToTensor() 

        ema_model.averaged_model.eval()
        
        while True:
            
            
            for itea_index in range(0, 100):  # itea_num
                print(itea_index)
                batch_npys = npy_names[itea_index * batch_size: (itea_index + 1) * batch_size]
                context = []
                x = []
                for npy_name in batch_npys:
                    con = torch.from_numpy(torch.load(samples_dir + npy_name)).unsqueeze(0)
                    context.append(con)
                    
                    x_temp = Image.open(images_dir+npy_name.replace('.npy', '.jpg'))
                    x_temp = transf(x_temp).unsqueeze(0)
                    new_size = Resize([128, 128])
                    x_temp = new_size(x_temp)
                    x.append(x_temp)
                context = torch.cat(context, 0).cuda()
                x = torch.cat(x, 0).cuda()
                
                x_visualisation = x.detach().clone()
                context_visualisation = context.detach().clone() if cfg.model.is_context_conditional else None

                if cfg.latent_diffusion:
                    
                    
                    with torch.no_grad():
                        sample_size = latent_encoder(x_visualisation[:1]).shape[-3:]
                else:
                    sample_size = x_visualisation.shape[-3:]
                
                self.generate_hard_identity(ema_model.averaged_model, size=sample_size,
                                                    x=x_visualisation, context=context_visualisation, latent_decoder=latent_decoder,
                                                    save_path=batch_npys,  N_PER_ROW=cfg.hard_per_num, gener_hard_path=cfg.gener_hard_path) 
                
            break


@hydra.main(config_path='configs', config_name='person_config', version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    trainer = DiffusionTrainerLite(devices='auto', accelerator='gpu', precision=cfg.training.precision)
    trainer.run(cfg)


if __name__ == "__main__":

    train()
