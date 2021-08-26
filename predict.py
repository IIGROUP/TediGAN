import argparse
import sys
import tempfile
from pathlib import Path

import cog

sys.path.insert(0, "base")

from utils.inverter import StyleGANInverter
from utils.visualizer import load_image, resize_image, save_image


class Predictor(cog.Predictor):
    def setup(self):
        # get default args first
        self.args = parse_args()

    @cog.input("image", type=Path, help="facial image for manipulation")
    @cog.input(
        "description",
        type=str,
        help="description of how to manipulate the image, e.g. 'he is old', 'she is smiling'",
    )
    def predict(self, image, description):

        self.args.description = description
        self.args.image_path = image

        inverter = StyleGANInverter(
            self.args.model_name,
            mode=self.args.mode,
            learning_rate=self.args.learning_rate,
            iteration=self.args.num_iterations,
            reconstruction_loss_weight=1.0,
            perceptual_loss_weight=self.args.loss_weight_feat,
            regularization_loss_weight=self.args.loss_weight_enc,
            clip_loss_weight=self.args.loss_weight_clip,
            description=self.args.description,
            logger=None,
        )
        image_size = inverter.G.resolution

        # Invert the given image.
        image = resize_image(
            load_image(str(self.args.image_path)), (image_size, image_size)
        )
        _, viz_results = inverter.easy_invert(image, num_viz=self.args.num_results)

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        save_image(str(out_path), viz_results[-1])
        return out_path


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="styleganinv_ffhq256",
        help="Name of the GAN model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="man",
        help="Mode (gen for generation, man for manipulation).",
    )
    parser.add_argument(
        "--description", type=str, default="he is old", help="The description."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="examples/142.jpg",
        help="Path of images to invert.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Directory to save the results. If not specified, "
        "`./results/inversion/test` "
        "will be used by default.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization. (default: 0.01)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=200,
        help="Number of optimization iterations. (default: 200)",
    )
    parser.add_argument(
        "--num_results",
        type=int,
        default=5,
        help="Number of intermediate optimization results to "
        "save for each sample. (default: 5)",
    )
    parser.add_argument(
        "--loss_weight_feat",
        type=float,
        default=5e-5,
        help="The perceptual loss scale for optimization. " "(default: 5e-5)",
    )
    parser.add_argument(
        "--loss_weight_enc",
        type=float,
        default=2.0,
        help="The encoder loss scale for optimization." "(default: 2.0)",
    )
    parser.add_argument(
        "--loss_weight_clip",
        type=float,
        default=2.0,
        help="The clip loss for optimization. (default: 2.0)",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="Which GPU(s) to use. (default: `0`)"
    )
    return parser.parse_args("")
