# Receptive Field in Computer Vision
- Reference: https://theaisummer.com/receptive-field/
- In [7], Luo et al. 2016 discover that not all pixels in a receptive field contribute equally to an output unit’s response. In the previous image, we observed that the receptive field varies with skip connections.
Obviously, the output feature is not equally impacted by all pixels within its receptive field. Intuitively, it is easy to perceive that pixels at the center of a receptive field have a much larger impact on output since they have more “paths” to contribute to the output. 

In short, the RF properties of the separable convolution are identical to its corresponding equivalent non-separable convolution. So, practically nothing changes in terms of the receptive field. 

As a natural consequence, one can define the relative importance of each input pixel as the effective receptive field (ERF) of the feature. In other words, ERF defines the effective receptive field of a central output unit as the region that contains any input pixel with a non-negligible impact on that unit. 

Specifically, as it is referenced in [7] we can intuitively realize the contribution of central pixels in the forward and backward pass as: 

“In the forward pass, central pixels can propagate information to the output through many different paths, while the pixels in the outer area of the receptive field have very few paths to propagate its impact. In the backward pass, gradients from an output unit are propagated across all the paths, and therefore the central pixels have a much larger magnitude for the gradient from that output

