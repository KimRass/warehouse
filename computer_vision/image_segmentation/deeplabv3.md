# Paper Summary
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587v3.pdf)
- ***With atrous convolution, one is able to control the resolution at which feature responses are computed within DCNNs without requiring learning extra parameters.***
- Employing large value of atrous rate enlarges the model's field-of-view, enabling object encoding at multiple scales.
## Related Works
- It has been shown that global features or contextual interactions are beneficial in correctly classifying pixels for semantic segmentation. In this work, we discuss four types of Fully Convolutional Networks (FCNs) that exploit context information for semantic segmentation.
- Image pyramid:
    - The same model, typically with shared weights, is applied to multi-scale inputs. ***Feature responses from the small scale inputs encode the long-range context, while the large scale inputs preserve the small object details.*** Typical examples include [...] who transform the input image through a Laplacian pyramid, feed each scale input to a DCNN and merge the feature maps from all the scales. [...] apply multi-scale inputs sequentially from coarse-to-fine, while [...] directly resize the input for several scales and fuse the features from all the scales.
- Encoder-decoder:
    - ***This model consists of two parts: (a) the encoder where the spatial dimension of feature maps is gradually reduced and thus longer range information is more easily captured in the deeper encoder output, and (b) the decoder where object details and spatial dimension are gradually recovered.*** For example, [...] employ deconvolution to learn the upsampling of low resolution feature responses. ***U-Net adds skip connections from the encoder features to the corresponding decoder activations,*** and [...] employs a Laplacian pyramid reconstruction network.
- Context module:
    - This model contains extra modules laid out in cascade to encode long-range context. One effective method is to incorporate DenseCRF [45] (with efficient high-dimensional filtering algorithms) to DCNNs.
- Spatial pyramid pooling:
    - This model employs spatial pyramid pooling to capture context at several ranges. Spatial pyramid pooling has also been applied in object detection.
- Atrous convolution:
    - Models based on atrous convolution have been actively explored for semantic segmentation.
## Methodology
- Our proposed module consists of atrous convolution with various rates and batch normalization layers which we found important to be trained as well.
- ***We discuss an important practical issue when applying a $3 \times 3$ atrous convolution with an extremely large rate, which fails to capture long range information due to image boundary effects, effectively simply degenerating to $1 \times 1$ convolution, and propose to incorporate image-level features into the ASPP module.***
- We duplicate several copies of the original last block in ResNet [32] and arrange them in cascade, and also revisit the ASPP module [11] which contains several atrous convolutions in parallel. Note that ***our cascaded modules are applied directly on the feature maps instead of belief maps.*** For the proposed modules, we experimentally find it important to train with batch normalization [38]. To further capture global context, we propose to augment ASPP with image-level features.
## References
- [11] [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/pdf/1412.7062v4.pdf)
- [32] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [39] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [45] [Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/pdf/1210.5644.pdf)

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
