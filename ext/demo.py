import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import math
import os
import numpy as np
import torch
import torchvision
from torch import optim
from tqdm import tqdm

from utils.common import CLIPLoss
from models.stylegan2.model import Generator
from base.models.perceptual_model import PerceptualModel
import clip

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str, required=True, default="man",
                      help="man for manipulation, gen for generation.")
  parser.add_argument("--description", type=str, required=True,
                      help="the text that guides the editing/generation")
  parser.add_argument("--ckpt", type=str, default="pretrained_models/stylegan2-ffhq-config-f.pt",
                      help="pretrained StyleGAN2 weights")
  parser.add_argument("--lr", type=float, default=0.1)
  parser.add_argument("--step", type=int, default=200,
                      help="number of optimization steps")
  parser.add_argument("--loss_pix_weight", type=float, default=1.0,
                      help='The pixel reconstruction loss scale for optimization. (default: 1.0)')
  parser.add_argument("--loss_reg_weight", type=float, default=2.0,
                      help='The latent loss scale for optimization. (default: 2.0)')
  parser.add_argument('--loss_feat_weight', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. (default: 5e-5)')
  parser.add_argument('--loss_clip_weight', type=float, default=1,
                      help='The clip loss for optimization. (default: 2.0)')
  parser.add_argument('--f_oom', type=bool, default=False,
                      help='if you have the Out-of-Memory problem, set as True. (default: False)')
  parser.add_argument("--latent_path", type=str, default='experiment/inference_results/inverted_codes.pt',
                      help="starts the optimization from the given latent code. Expects a .pt format")
  parser.add_argument("--save_intermediate_image_every", type=int, default=20,
                      help="if > 0 then saves intermidate results during the optimization")
  parser.add_argument("--results_dir", type=str, default="experiment")
  return parser.parse_args()

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp

def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()

def main(args):
    
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)

    F = PerceptualModel(min_val=-1.0, max_val=1.0)

    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    z_mean = g_ema.mean_latent(4096)
    # z_load = np.load(args.latent_path)
    # z_init = torch.from_numpy(z_load).cuda()
    # print(np.shape(latent_load))
    F_OOM = args.f_oom

    if args.mode =="man":
        z_init = torch.load(args.latent_path).cuda()
    else:
        z_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, z_init = g_ema([z_init_not_trunc], truncation_latent=z_mean, return_latents=True,
                              truncation=0.7)

    x, _ = g_ema([z_init], input_is_latent=True, randomize_noise=False)

    # z = z_init.detach().clone()
    z = z_mean.detach().clone().repeat(1, 18, 1)

    z.requires_grad = True

    clip_loss = CLIPLoss()

    optimizer = optim.Adam([z], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        x_rec, _ = g_ema([z], input_is_latent=True, randomize_noise=False)
        if not F_OOM:
            loss = 0.0
            # Reconstruction loss.
            loss_pix = torch.mean((x - x_rec) ** 2)
            loss = loss + loss_pix * args.loss_pix_weight
            log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

            # Perceptual loss.
            if args.loss_feat_weight:
                x_feat = F.net(x)
                x_rec_feat = F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
                loss = loss + loss_feat * args.loss_feat_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            # Regularization loss.
            if args.loss_reg_weight:
                loss_reg = torch.mean((z_init - z) ** 2)
                # loss_reg = ((z_init - z) ** 2).sum()
                loss = loss + loss_reg * args.loss_reg_weight
                log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'

            # CLIP loss.
            if args.loss_clip_weight:
                loss_clip = clip_loss(x_rec, text_inputs)
                loss = loss + loss_clip[0][0] * args.loss_clip_weight
                log_message += f', loss_clip: {_get_tensor_value(loss_clip[0][0]):.3f}'
        else:
            loss_reg = ((z_init - z) ** 2).sum()
            loss_clip = clip_loss(x_rec, text_inputs)
            loss = loss_reg + loss_clip[0][0] * args.loss_clip_weight # set loss_clip_weight as 200 in my case.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description((f"loss: {loss.item():.4f};"))

    final_result = torch.cat([x, x_rec])
    return final_result

if __name__ == "__main__":
    args = parse_args()
    result_image = main(args)
    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(
        args.results_dir, "final_result.png"), normalize=True, scale_each=True, range=(-1, 1))
