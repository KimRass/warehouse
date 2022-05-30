# `gym`
```python
import gym
```
## `gym.make()`
## `gym.envs`
### `gym.envs.toy_text`
#### `gym.envs.toy_text.discrete`
### `env.nS`
### `env.nA`
### `env.P_tensor`
- Return state transition matrix.
### `env.R_tensor`
- Return reward.
### `env.reset()`
### `env.observation_space`
### `env.action_space`
#### `env.action_space.sample()`
### `env.step()`
```python
next_state, rew, done, info = env.step(act)
```
### `env.render()`

# `io`
## `BytesIO`
```python
from io import BytesIO
```
```python
url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/img/POSTER_20M_FULL/{}.jpg".format(movie_id)
req = requests.get(url)
b = BytesIO(req.content)
img = np.asarray(Image.open(b))
```