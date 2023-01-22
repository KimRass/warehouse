- Source: https://medium.com/jun94-devpblog/dl-12-unsampling-unpooling-and-transpose-convolution-831dc53687ce

# Unpooling
## Nearst Neighbor
- ![nearst_neighbor](https://miro.medium.com/max/1364/1*9N9FVYalaVAk-aalcBVYaA.webp)
- Nearest-Neighbor is the simplest approach to upsample. It copies a pixel value (response) of input feature map to all pixels in the corresponding sub-region of output.
- Although its simplicity, the problem of this approach is that the output structure becomes blocky as all pixels in each subregion have the same value.
## Bed of Nails
- ![bed_of_nails](https://miro.medium.com/max/1384/1*LQVVqK8YJ4ndcU6NS4UOxA.webp)
- Another de-pooling method, the Bed-of-Nails operation put an input feature response (element) in the top-left corner of the corresponding sub-region of de-pooling output, and it sets all the other elements in the sub-region to zero.
- By doing so, it achieves the fine-grained output structure. However, the upsampled elements always have a fixed location.
## Max Unpooling
- ![max_unpooling](https://miro.medium.com/max/1400/1*b0NUJ-7IJnrljrzAc07BzQ.webp)
- In order to supplement the problem of Bed of Nails, Max Unpooling is introduced. While Max Unpooling performs upsampling in a similar manner as Bed of Nails, it remembers the indices of where the largest elements come from before max pooling. And this information is used later on when Max Unpooling is performed to place the elements in the positions of each sub-region where they are previously located before max pooling.

# Transpose Convolution
- We have taken a look at upsampling approaches based on unpooling. As one might notice, the previously mentioned three methods are fixed numerical equations (functions) and there is no learning taking place.
- Another type of upsampling, which makes use of learning, is called transpose convolution.
- ![transpose_convolution](https://miro.medium.com/max/1400/1*Cti98FtmLNJEgWdQ0HorRQ.webp)