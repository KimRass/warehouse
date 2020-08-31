# gym
```python
import gym
```
## gym.make()
```python
env = gym.make("Taxi-v1")
```
### env.reset()
```python
observ = env.reset()
```
### env.render()
### env.action_space.sample()
```python
action = env.action_space.sample()
```
### env.step()
```python
observ, reward, done, info = env.step(action)
```
```python
env = gym.make("Taxi-v1")
observ = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observ, reward, done, info = env.step(action)
```
