# Paper Summary
- [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles, 2017](https://arxiv.org/pdf/1603.09246.pdf)
## Introduction
- We introduce a novel self-supervised task, the Jigsaw puzzle reassembly problem, which builds features that yield high performance when transferred to detection and classification tasks. ***We argue that solving Jigsaw puzzles can be used to teach a system that an object is made of parts and what these parts are.*** The association of each separate puzzle tile to a precise object part might be ambiguous. However, when all the tiles are observed, the ambiguities might be eliminated more easily because the tile placement is mutually exclusive. Training a Jigsaw puzzle solver takes about 2.5 days compared to 4 weeks of [10].
- This work falls in the area of representation/feature learning, which is an unsupervised learning problem. Representation learning is concerned with building intermediate representations of data useful to solve machine learning tasks.
- It also involves transfer learning, as one applies and repurposes features that have been learned by solving the Jigsaw puzzle to other tasks such as object classification and detection. In our experiments we do so via the pre-training + fine-tuning scheme, as in prior work. Pre-training corresponds to the feature learning that we obtain with our Jigsaw puzzle solver. Fine-tuning is instead the process of updating the weights obtained during pre-training to solve another task (object classification or detection).
- In this work we design features to separate the appearance from the arrangement (geometry) of parts of objects.
- Training the Jigsaw puzzle solver also amounts to building a model of both appearance and configuration of the parts.
- Figure 1
    - <img src="https://user-images.githubusercontent.com/105417680/227456353-63cacce7-009c-481b-9665-a274551eb4b4.png" width="500">
    - (b) A puzzle obtained by shuffling the tiles. ***Some tiles might be directly identifiable as object parts, but others are ambiguous (e.g., have similar patterns) and their identification is much more reliable when all tiles are jointly evaluated.***
    - ***In contrast, with reference to (c), determining the relative position between the central tile and the top two tiles from the left can be very challenging [10].***
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/227456305-d98a5faf-c058-4066-b020-5ba565878e35.png" width="500">
    - Most of the shape of these 2 pairs of images is the same (two separate instances within the same categories). However, ***some low-level statistics are different (color and texture). The Jigsaw puzzle solver learns to ignore such statistics when they do not help the localization of parts.***
- An important factor to learn better representations is to prevent our model from taking the undesirable solutions to solve the pre-text task. We call these solutions shortcuts. ***Other shortcuts that the model can use to solve the Jigsaw puzzle task include exploiting low-level statistics, such as edge continuity, the pixel intensity/color distribution, and chromatic aberration.***
## Related Works
- Self-superversied learning
    - Recently, [10], Wang and Gupta [39] and Agrawal et al. have explored a novel paradigm for unsupervised learning called self-supervised learning. ***The main idea is to exploit different labelings that are freely available besides or within visual data, and to use them as intrinsic reward signals to learn general-purpose features.***
    - This learning strategy is a recent variation on the unsupervised learning theme that exploits labeling that comes for "free" with the data.
- [10]
    - Uses the relative spatial co-location of patches in images as a label.
    - Trains a convolutional network to classify the relative position between two image patches. One tile is kept in the middle of a 3 × 3 grid and the other tile can be placed in any of the other 8 available locations (up to some small random shift). There are some examples where the relative location between the central tile and the top-left and top-middle tiles is ambiguous. ***In contrast, the Jigsaw puzzle problem is solved by observing all the tiles at the same time.***
## Methodology
- ***Low-level statistics, such as similar structural patterns and texture close to the boundaries of the tile, are simple cues that humans actually use to solve Jigsaw puzzles. However, solving a Jigsaw puzzle based on these cues does not require any understanding of the global object. Thus, here we present a network that delays the computation of statistics across different tiles.***
- Shortcuts
    - To avoid shortcuts we employ multiple techniques. ***We make sure that the tiles are shuffled as much as possible by choosing configurations with sufficiently large average Hamming distance.*** In this way the same tile would have to be assigned to multiple positions (possibly all 9) thus making the mapping of features $F_{i}$ to any absolute position equally likely.
    - ***To avoid shortcuts due to edge continuity and pixel intensity distribution we also leave a random gap between the tiles. This discourages the CFN from learning low-level statistics.***
    - To avoid shortcuts due to chromatic aberration we jitter the color channels and use grayscale images
- Permutation set
    - We generate the permutation set iteratively via a greedy algorithm. We begin with an empty permutation set and at each iteration select the one that has the desired Hamming distance to the current permutation set. ***For the minimal and middle case, the*** $\argmax_{k}$ ***function at line 10 is replaced by*** $\argmin_{k}$ ***and uniform sampling respectively. Note that the permutation set is generated before training.***
    - To increase the minimum possible distance between permutations, we remove similar permutations in a maximal set with 100 initial entries.
- Preventing Shortcuts
    - Table 5: The performance of transfer learning the CFN to the detection task on Pascal VOC.
        - <img src="https://user-images.githubusercontent.com/105417680/227773531-d9b67d0a-ef54-402b-932b-8ba79f592c8a.png" width="800">
        - ***In a self-supervised learning method, shortcuts exploit information useful for solving the pre-text task, but not for a target task, such as detection.*** We experimentally show that the CFN can take the following shortcuts to solve the Jigsaw Puzzle task:
            - Edge continuity
                - ***'Gap': A strong cue to solve Jigsaw puzzles is the continuity of edges. We select the 64 × 64 pixel tiles randomly from the 85 × 85 pixel cells. This allows us to have a 21 pixel gap between tiles.***
            - Low level statistics
                - Adjacent patches include similar low-level statistics like the mean and standard deviation of the pixel intensities. This allows the model to find the arrangement of the patches.
                - ***'Normalization': To avoid this shortcut, we normalize the mean and the standard deviation of each patch independently.***
            - Chromatic Aberration
                - Chromatic aberration is a relative spatial shift between color channels that increases from the images center to the borders. This type of distortion helps the network to estimate the tile positions.
                - ***'Color jittering': To avoid this shortcut, we use three techniques: i) We crop the central square of the original image and resize it to 255 × 255; ii) We train the network with both color and grayscale images. Our training set is a composition of grayscale and color images with a ratio of 30% to 70%; iii) We (spatially) jitter the color channels of the color images of each tile randomly by ±0, ±1, ±2 pixels.***
## Architecture
- Figure 3: Context Free Network
    - <img src="https://user-images.githubusercontent.com/105417680/227456473-db45c069-2a93-4adf-b340-35743a1863c0.png" width="1100">
    - ***During training we resize each input image until either the height or the width matches 256 pixels and preserve the original aspect ratio then the other dimension is cropped to 256 pixels. (Comment: 가로와 세루 중 더 작은 쪽을 256에 맞춰 리사이즈한 다음 다른 한쪽을 256에 맞춰 자른다는 얘기입니다.) Then, we crop a random region from the resized image of size 225 × 225 and split it into a 3 × 3 grid of 75 × 75 pixels tiles. We then extract a 64 × 64 region from each tile by introducing random shifts and feed them to the network Thus, we have an average gap of 11 pixels between the tiles. However, the gaps may range from a minimum of 0 pixels to a maximum of 22 pixels.***
    - The network first computes features based only on the pixels within each tile (one row in the figure). Then, it finds the parts arrangement just by using these features (last fully connected layers in the figure). The objective is to force the network to learn features that are as representative and discriminative as possible of each object part for the purpose of determining their relative location.
    - We build a convolutional network, where each row up to the first fully connected layer (fc6) uses the AlexNet architecture [25] with shared weights. (Comment: AlexNet과 완전히 같지는 않고 각 convolution 레이어의 채널 수는 서로 조금 다릅니다.) ***The outputs of all fc6 layers are concatenated and given as input to fc7. All the layers in the rows share the same weights up to and including fc6. We call this architecture the context-free network (CFN) because the data flow of each patch is explicitly separated until the fully connected layer and context is handled only in the last fully connected layers.***
    - For simplicity, we do not indicate the maxpooling and ReLU layers. These shared layers are implemented exactly as in AlexNet [25].
    - ***The 9 tiles are reordered via a randomly chosen permutation from a predefined permutation set*** and are then fed to the CFN. The task is to predict the index of the chosen permutation (technically, we define as output a probability vector with 1 at the 64-th location and 0 elsewhere).
    - ***Notice instead, that during the training on the puzzle task, we set the stride of the first layer of the CFN to 2 instead of 4.***
## Training
### Pretext Task
- ***To train the CFN we define a set of Jigsaw puzzle permutations, e.g., a tile configuration*** $S = (3, 1, 2, 9, 5, 4, 8, 7, 6)$, ***and assign an index to each entry. We randomly pick one such permutation, rearrange the 9 input patches according to that permutation, and ask the CFN to return a vector with the probability value for each index.*** Given 9 tiles, there are $9! = 362,880$ possible permutations. From our experimental validation, we found that the permutation set is an important factor on the performance of the representation that the network learns.
- The output of the CFN can be seen as the conditional probability density function (pdf) of the spatial arrangement of object parts (or scene parts) in a part-based model, i.e.,
### Transfer learning
- In the transfer learning experiments we show results with the trained weights transferred on AlexNet (precisely, stride 4 on the first layer). The training in the transfer learning experiment is the same as in the other competing methods.
- We verify that this architecture performs as well as AlexNet in the classification task on the ImageNet 2012 dataset. ***In this test we resize the input images to 225 × 225 pixels, split them into a 3 × 3 grid and then feed the full 75 × 75 tiles to the network.*** We find that the CFN achieves 57.1% top-1 accuracy while AlexNet achieves 57.4% top-1 accuracy. However, the CFN architecture is more compact than AlexNet. It depends on only 27.5M parameters, while AlexNet uses 61M parameters. The fc6 layer includes 4 × 4 × 256 × 512 ∼ 2M parameters while the fc6 layer of AlexNet includes 6 × 6 × 256 × 4096 ∼ 37.5M parameters. However, the fc7 layer in our architecture includes 2M parameters more than the same layer in AlexNet. This network can thus be used interchangeably for different tasks including detection and classification.
- Pascal VOC 2007 classification and detection and PASCAL VOC 2012 segmentation
    - Table 1: Self-supervised learning with Jigsaw puzzle vs. Supervised learning with AlexNet [25]
        - <img src="https://user-images.githubusercontent.com/105417680/227773423-3cf16914-f035-4e06-87ee-5b9265a870c2.png" width="800">
        - We fine-tune the Jigsaw task features on the classification task on PASCAL VOC 2007 by using the framework of [24] and on the object detection task by using the Fast R-CNN [16] framework. We also fine-tune our weights for the semantic segmentation task using the framework [27] on the PASCAL VOC 2012 dataset.
        - Because our fully connected layers are different from those of the standard AlexNet, we select one row of the CFN (up to conv5), copy only the weights of the convolutional layers, and fill the fully connected layers with Gaussian random weights with mean 0.1 and standard deviation 0.001.
        - Our features outperformed all other methods and closed the gap with features obtained with supervision.
- ImageNet 2012 Classification
    - Table 2: Jigsaw puzzle $\rightarrow$ ImageNet 2012 classification
        - <img src="https://user-images.githubusercontent.com/105417680/227773444-79251b71-09d7-40a2-ad97-1cbbadb0b131.png" width="600">
        - 'CFN', '[10]' and '[39]': Training each network with the labeled data from ImageNet 2012 by locking a subset of the layers
        - 'Random': Initializing the unlocked layers with random values.
        - The numbers are obtained by averaging 10 random crops predictions.
        - If we train AlexNet, we obtain the reference maximum accuracy of 57.4%.
        - 'conv5': Our method achieves 34.6% when only fully connected layers are trained.
        - 'conv4': There is a significant improvement (from 34.6% to 45.3%) when the conv5 layer is also trained. ***This shows that the conv5 layer starts to be specialized on the Jigsaw puzzle reassembly task.***
    - Table 3: ImageNet 2012 classification $\rightarrow$ Jigsaw puzzle
        - <img src="https://user-images.githubusercontent.com/105417680/227773468-16734191-6b8e-458d-b2f1-fbf84a9d174e.png" width="600">
        - We also perform a novel experiment to understand ***whether semantic classification is useful to solve Jigsaw puzzles, and thus to see how much object classification and Jigsaw puzzle reassembly tasks are related.***
        - We take the pretrained AlexNet and transfer its features to solve Jigsaw puzzles. We also use the same locking scheme to see the transferability of features at different layers. Compared to the maximum accuracy of the Jigsaw task, 88%, we can see that semantic training is quite helpful towards recognizing object parts. Indeed, the performance is very high up to conv4. (Comment: conv4까지의 모든 레이어들을 freeze하고 conv5만 jigsaw puzzle에 대해서 transfer learning했을 때 jigsaw puzzle 성능이 83% ~ 88%로 매우 높습니다. 즉 conv4까지의 모든 레이어들은 ImageNet classification에 대해 훈련시키든 아니면 jigsaw puzzle에 대해 훈련시키든 representation learning은 크게 달라지지 않음을 알 수 있습니다. conv5는 두 경우에 대해서 서로 다른 represnetation learning을 보입니다.)
## Experiments
- Permutation Set
    - Table 4: Ablation study on the permutation set.
        - <img src="https://user-images.githubusercontent.com/105417680/227516962-497a199e-a7bc-4fe4-8784-021e34cde898.png" width="600">
        - The permutation set controls the ambiguity of the task. ***If the permutations are close to each other, the Jigsaw puzzle task is more challenging and ambiguous. For example, if the difference between two different permutations lies only in the position of two tiles and there are two similar tiles in the image, the prediction of the right solution will be impossible.***
        - To show this issue quantitatively, we compare the performance of the learned representation on the PASCAL VOC 2007 detection task by generating several permutation sets based on the following three criteria:
            - Cardinality (Number of permutations)
                - ***We find that as the total number of permutations increases, the training on the Jigsaw task becomes more and more difficult.*** (Comment: 'Number of permutations'가 증가함에 따라 'Jigsaw task accuracy가 감소합니다.')
                - ***Also, we find that the performance of the detection task increases with a growing number of permutations.*** (Comment: 'Number of permutations'가 증가함에 따라 'Detection performance'가 감소합니다.')
            - Average Hamming distance
                - ***One can see that the average Hamming distance between permutations controls the difficulty of the Jigsaw puzzle reassembly task, and it also correlates with the object detection performance. We find that as the average Hamming distance increases, the CFN yields lower Jigsaw puzzle solving errors and lower object detection errors with fine-tuning.*** (Comment: 'Average hamming distance'가 증가함에 따라 'Jigsaw task accuracy'와 'Detection performance'가 모두 증가합니다.)
                - We can see that large Hamming distances are desirable.
            - Minimum hamming distance
                - ***The minimum distance helps to make the task less ambiguous. The best performing permutation set is a trade off between the number of permutations and how dissimilar they are from each other.*** (Comment: 'Number of permutations'가 증가함에 따라 그리고 'Minimum hamming distance'가 감소함에 따라 'Detection performance'가 증가합니다. 큰 'Minimum hamming distance'는 서로 다른 permutation이 서로 유사하지 않다('dissimilar')는 것을 의미합니다. 서로 유사하지 않다는 것은 모호하지 않다(not 'ambiguous')는 말입니다.)
            - The outcome of this ablation study seems to point to the following final consideration: A good self-supervised task is neither simple nor ambiguous. (Comment: 'Detection performance'는 'Number of permutations'가 증가할수록(not 'simple'), 'Minumum hamming distance'가 감소할수록('ambiguous') 증가합니다. 따라서 Table 4를 가지고 내릴 수 있는 결론은 "A good self-supervised task is either difficult or ambiguous."라고 생각합니다.)
## References
- [10] [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192.pdf)
- [16] [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- [25] [ImageNet Classification with Deep Convolutional Neural Networks, 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [27] [Fully Convolutional Networks for Semantic Segmentation, 2015](https://arxiv.org/pdf/1411.4038.pdf)

- We use stochastic gradient descent without batch normalization [21] on one Titan X GPU. The training uses 1.3M color images of 256×256 pixels and mini- batches with a batch size of 256 images. The images are resized by preserving the aspect ratio until either the height or the width matches 256 pixels. Then the other dimension is cropped to 256 pixels. The training converges after 350K iterations with a basic learning rate of 0.01 and takes 59.5 hours in total (∼ 2.5 days). If we take 122% = 3072cores@1000Mhz 2880cores@875Mhz = 6,144GFLOPS 5,040GFLOPS as the best possible performance ratio between the Titan X and the Tesla K40 (used for [10]) we can predict that the CFN would have taken ∼ 72.5 hours (∼ 3 days) on a Tesla K40. We compute that on average each image is used 350K × 256/1.3M ' 69 times. That is, we solve on average 69 Jigsaw puzzles per image.
