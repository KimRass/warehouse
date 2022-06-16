# DCGAN (Deep Convolutional GAN) 
# cGAN (Conditional GAN)
- The conditional generative adversarial network, or cGAN for short, is an extension to the GAN architecture that makes use of information in addition to the image as input both to the generator and the discriminator models. For example, if class labels are available, they can be used as input. 
# Pix2Pix 
- The pix2pix model is an extension of the GAN for image-conditional image generation, referred to as the task image-to-image translation. A U-Net model architecture is used in the generator model, and a PatchGAN model architecture is used as the discriminator model 
# StyleGAN
- Source: https://www.google.com/amp/s/neptune.ai/blog/6-gan-architectures/amp 
- StyleGAN is a GAN formulation which is capable of generating very high-resolution images even of 1024*1024 resolution. The idea is to build a stack of layers where initial layers are capable of generating low-resolution images (starting from 2\ *2) and further layers gradually increase the resolution.