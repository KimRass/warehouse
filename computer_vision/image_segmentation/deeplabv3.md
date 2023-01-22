- Reference: https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74
- DeepLabv3 outperforms DeepLabv1 and DeepLabv2, even with the post-processing step Conditional Random Field (CRF) removed, which is originally used in DeepLabv1 and DeepLabv2.

# Dilation (Atrous) Convolution
- ![atrous_convolution](https://miro.medium.com/max/1130/1*-r7CL0AkeO72MIDpjRxfog.webp)
# Dilation Convolution using Multi-Grid
- !["](https://miro.medium.com/max/1400/1*nFJ_GqK1D3zKCRgtnRfrcw.webp)

# Depth-wise Separable Convolution
- Reference: https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
- Normal convolution layer of a neural network involve (input_channel * kernel_width * kernel_height * output_channel) parameters. Having too much parameters increases the chance of over-fitting.
## Normal Convolution
- In this convolution, we apply a 2-d depth filter at each depth level of input tensor. Lets understand this through an example. Suppose our input tensor is 8x8x3. Filter is 3x3x3. In a standard convolution we would directly convolve in depth dimension as well. You would require three 3-channel filters if you would use normal convolution.
- Number of parameters: 3 * (3x3x3) = 81
- ![normal_convolution](https://miro.medium.com/max/828/1*sYpl-7LlrtlOKW8RvlyKOg.png)
## Depth-wise Convolution
- In depth-wise convolution, we use each filter channel only at one input channel. In the example, we have 3 channel filter and 3 channel image. What we do is — break the filter and image into three different channels and then convolve the corresponding image with corresponding channel and then stack them back.
- ![depthwise_convolution](https://miro.medium.com/max/1100/1*Esdvt3HLoEQFen94x29Z0A.png)
- To produce same effect with normal convolution, what we need to do is - select a channel, make all the elements zero in the filter except that channel and then convolve. We will need three different filters — one for each channel.
- Number of parameters: 3 * (3x3) = 27
## Depth-wise Separable Convolution
- We perform depth-wise convolution and after that we use a 1x1 filter to cover the depth dimension (point-wise convolution).
- ![depthwise_separable_convolution](https://miro.medium.com/max/1100/1*JwCJCgN2UreEn3U1nwVj8Q.png)
- One thing to notice is, how much parameters are reduced by this convolution to output same no. of channels.
- Number of parameters: 3 * (3x3) + 3 * (1x3) = 36
- Having too many parameters forces function to memorize lather than learn and thus over-fitting. Depth-wise separable convolution saves us from that.
```python
class DepthwiseSeperableConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, groups=in_dim)
        self.pointwise = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        
    def forward(self, input):
        z = self.depthwise(input)
        output = self.pointwise(z)
        return output
```