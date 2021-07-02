# python 3.6
"""Revised from Inverts given images to latent codes with In-Domain GAN Inversion."""

import os
import argparse

from utils.inverter import StyleGANInverter
from utils.visualizer import save_image, load_image, resize_image

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str,
                      default='styleganinv_ffhq256', help='Name of the GAN model.')
  parser.add_argument('--mode', type=str,
                      default='man', help='Mode (gen for generation, man for manipulation).')
  parser.add_argument('--description', type=str, default='he is old',
                      help='The description.')
  parser.add_argument('--image_path', type=str, default='examples/142.jpg', help='Path of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/test` '
                           'will be used by default.')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=200,
                      help='Number of optimization iterations. (default: 200)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--loss_weight_clip', type=float, default=2.0,
                      help='The clip loss for optimization. (default: 2.0)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.isfile(args.image_path)
  output_dir = args.output_dir or f'results/inversion/test'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  inverter = StyleGANInverter(
      args.model_name,
      mode=args.mode,
      learning_rate=args.learning_rate,
      iteration=args.num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=args.loss_weight_feat,
      regularization_loss_weight=args.loss_weight_enc,
      clip_loss_weight=args.loss_weight_clip,
      description=args.description,
      logger=None)
  image_size = inverter.G.resolution

  # Invert the given image.
  image = resize_image(load_image(args.image_path), (image_size, image_size))
  _, viz_results = inverter.easy_invert(image, num_viz=args.num_results)

  if args.mode == 'man':
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
  else:
    image_name = 'gen'
  save_image(f'{output_dir}/{image_name}_ori.png', viz_results[0])
  save_image(f'{output_dir}/{image_name}_enc.png', viz_results[1])
  save_image(f'{output_dir}/{image_name}_inv.png', viz_results[-1])
  print(f'save {image_name} in {output_dir}')

if __name__ == '__main__':
  main()
