import os
import numpy as np
from PIL import Image
import streamlit as st
import torch
import clip

from utils.inverter import StyleGANInverter
from utils.visualizer import resize_image

st.title("Text-Guided Editing of Images")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

mode = st.selectbox('Mode', ('gen', 'man'))

description = st.text_input('Description', 'she is young')

n_step = st.slider('Step (10*n)', min_value=0, max_value=100, value=20)

n_lr = st.slider('Learning Rate (10^-n)',
                 min_value=1, max_value=5, value=2)

lambda_clip = st.slider('The clip loss weight for optimization', min_value=0.0,
                      max_value=10.0, value=1.0, step=0.01)
lambda_feat = st.slider('The perceptual loss weight for optimization', min_value=0.0,
                        max_value=10.0, value=2.0, step=1.0)
lambda_l2 = st.slider('The reconstruction loss weight for optimization', min_value=0.0,
                        max_value=10.0, value=1.0)
lambda_enc = st.slider('The encoding loss weight for optimization (10^-n)', min_value=0,
                       max_value=10, value=5, step=1)
step = 10*n_step
ini_lr = 10**(-n_lr)

st.write("total step is", step, ', the learning rate is', ini_lr)

model_name = 'styleganinv_ffhq256'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
  """Main function."""

  inverter = StyleGANInverter(model_name, 
      mode=mode,
      learning_rate=ini_lr,
      iteration=step,
      reconstruction_loss_weight=lambda_l2,
      perceptual_loss_weight=lambda_feat,
      regularization_loss_weight=lambda_enc,
      clip_loss_weight=lambda_clip,
      description=description)
  image_size = inverter.G.resolution

  text_inputs = torch.cat([clip.tokenize(description)]).cuda()

  # Invert images.
  # uploaded_file = uploaded_file.read()
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    # st.write("")
    st.write("Just a second...")

    image = resize_image(np.array(image), (image_size, image_size))
    _, viz_results = inverter.easy_invert(image, 1)
    if mode=='man':
      final_result = np.hstack([image, viz_results[-1]])
    else:
      final_result = np.hstack([viz_results[1], viz_results[-1]])


    # return final_result
    with st.beta_container():
        st.image(final_result, use_column_width=True)

if __name__ == '__main__':
  main()
