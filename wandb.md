# Install
```sh
!pip install -q --upgrade wandb
```
```python
import wandb
from wandb.keras import WandbCallback

wandb.login()

# wandb config
WANDB_CONFIG = {
    "competition": "AG News Classification Dataset", 
    "_wandb_kernel": "neuracort"
}

run = wandb.init(project='ag-news', config=WANDB_CONFIG)

callbacks = [..., WandbCallback()]
...

model.fit(..., callbacks=callbacks)

wandb.finish()
```