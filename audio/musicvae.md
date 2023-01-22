- Reference: https://magenta.tensorflow.org/music-vae
- These desires have led us to focus much of our recent efforts on what are known as latent space models. The technical goal of this class of models is to represent the variation in a high-dimensional dataset using a lower-dimensional code, making it easier to explore and manipulate intuitive characteristics of the data. As a creative tool, the goal is to provide an intuitive palette with which a creator can explore and manipulate the elements of an artistic work.
- Latent space models are capable of learning the fundamental characteristics of a training dataset and can therefore exclude these unconventional possibilities.
- Aside from excluding unrealistic examples, latent spaces are able to represent the variation of real data in a lower-dimensional space. This means that they can also reconstruct real examples with high accuracy. Furthermore, when compressing the space of the dataset, latent space models tend to organize it based on fundamental qualities, which clusters similar examples close together and lays out the variation along vectors defined by these qualities.
- The desirable properties of a latent space can be summarized as follows:
Expression: Any real example can be mapped to some point in the latent space and reconstructed from it.
Realism: Any point in this space represents some realistic example, including ones not in the training set.
Smoothness: Examples from nearby points in latent space have similar qualities to one another.
- Realism allows you to randomly sample new examples that are similar to those in your dataset by decoding from a randomly selected point in latent space.
- Autoencoder: One limitation with this type of autoencoder is that it often has “holes” in its latent space. This means that if you decoded a random vector, it might not result in anything realistic. For example, NSynth is capable of reconstruction and interpolation, but it lacks the realism property, and thus the ability to randomly sample, due to these holes.

- Notice that the intermediate sequences are now valid, and the transitions between them are smooth. The intermediate sequences are also not restricted to selecting from the notes in the originals as before, and yet the note selection makes more musical sense in the context of the endpoints. In this example we fully satisfy properties of expression, realism and smoothness.

- For short sequences (e.g., 2-bar "loops"), we use a bidirectional LSTM encoder
and LSTM decoder. For longer sequences, we use a novel hierarchical LSTM
decoder, which helps the model learn longer-term structures.

# Paper Review
- Paper: [A Hierarchical Latent Vector Model
for Learning Long-Term Structure in Music](https://arxiv.org/pdf/1803.05428.pdf)
- existing recurrent VAE models have difficulty modeling sequences with long-term structure. To address this issue, we propose the use of a hierarchical decoder, which first outputs embeddings for subsequences of the input and then uses these embeddings to generate each subsequence independently. This structure encourages the model to utilize its latent code, thereby avoiding the “posterior collapse” problem, which remains an issue for recurrent VAEs.
## Introduction
- 생성 모델은 데이터 x를 생성하는 데 쓰이는 확률 분포 p를 예측하는 것입니다. 이것은 새로운 Datapoints를 샘플링하는 것을 가능하게 만듭니다.
- 이 논문에서는 VAE와 같은 Deep latent variable models에 주목합니다.
- z: Latent space 상의 분포 또는 실제 데이터로부터 샘플링된 Latent vector
- Attribute vector
    - 어떤 Attribute를 가진 모든 Datapoints의 Latent codes를 평균하여 생성됩니다.
    - 예를 들어 갈색 머리를 가진 사람의 사진을 Encoding하고 갈색 머리에 해당하는 Attribute vector를 빼고 금발에 해당하는 Attribute vecotr를 더한 뒤 이를 Decoding하면 금발을 가진 사람의 사진을 만들 수 있다는 것입니다.
    - 서로 다른 Latent vectors를 Interpolate하여 중간적인 Attribute를 가진 사람의 사진을 생성하는 것도 가능합니다.
- Autoencoder는 짧은 Sequences에 대해서는 잘 작동하시만 매우 긴 Sequences에 대해서는 Deep latent variable models가 활용될 여지가 있습니다.
- 이 논문에서 제시하는 모델은 Hierarchical recurrent decoder를 갖는 새로운 Sequential autoencoder입니다. Sequence의 전체를 하나의 Latent vector로 Encode합니다. 긴 Sequences에 대해서 일반적인 RNN Decoder보다 더 성능이 우수함을 실험적으로 보일 것입니다.
## Background
- Fundamentally, our model is an autoencoder, i.e., its goal is to accurately reconstruct its inputs. However, we additionally desire the ability to draw novel samples and perform latent-space interpolations and attribute vector arithmetic. For these properties, we adopt the framework of the Variational Autoencoder.
### Variational Autoencoders
- A common constraint applied to autoencoders is that they compress the relevant information about the input into a lower-dimensional latent code. Ideally, this forces the model to produce a compressed representation that captures important factors of variation in the dataset.
