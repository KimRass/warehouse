# `datasets`
## Load Dataset from Hub
- Reference: https://huggingface.co/docs/datasets/loading
```python
from datasets import Dataset, load_from_disk, load_metric, Features, Audio, Value

# Some datasets may have more than one version based on Git tags, branches, or commits. Use the revision parameter to specify the dataset version you want to load.
# `split`: e.g., `"train+test"`, `"train[10:20]"`, `"train[:10%]"`
load_dataset(path, name, split, [cached_dir])
# Example
ds = load_dataset(path="kresnik/zeroth_korean", name="clean", split="test")
```
## Load Dataset from Local
```python
# `path`: (`"csv"`, `"json"`, `"text"`)
load_dataset(path, data_files)
```
```python
ds.features
ds.column_names
ds.remove_columns(["col1", "col2", ...])
```
## Save or Load
```python
ds.save_to_dist("...")
ds = load_from_disk("...")

ds.to_csv()
ds.to_json()
ds.to_pandas()
ds.to_dict()
```
## Sort
```python
ds.sort("<feature>")
```
## Shffule
```python
ds.shuffle([seed])
```
## Split
```python
ds.train_test_split(test_size, [shuffle=True])
```
## Sample
```python
# Filter rows according to a list of indices.
ds.select()
```
## Shard
```python
ds.shard(num_shard, index)
```
## Map?
```python
# `batched=True`: Batched mode. In batched mode `datasets.Dataset.map()` will provide batch of examples (as a dict of lists) to the mapped function and expect the mapped function to return back a batch of examples (as a dict of lists) but the input and output batch are not required to be of the same size.
# `num_proc`: Number of processes for multiprocessing. By default it doesn’t use multiprocessing.
ds.map([remove_columns], [batched], num_proc)
```

# `transformers`
```python
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
```
## `TrainingArguments`
```python
# `fp16`: Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
# `group_by_length`: Whether or not to group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient). Only useful if applying dynamic padding.

# * Number of steps: Number of samples / Batch size
# `gradient_accumulation_steps`: Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
# `evaluation_strategy`: The evaluation strategy to adopt during training.
	# `"no"`: No evaluation is done during training.
	# `"steps"`: Evaluation is done (and logged) every `eval_steps`.
	# `"epoch"`: Evaluation is done at the end of each epoch.
# `eval_steps`: Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same value as `logging_steps` if not set.
# `warmup_steps`: Number of steps used for a linear warmup from 0 to `learning_rate`.
# `save_total_limit`: If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`.
args_tr = TrainingArguments()
trainer = Trainter(
	...
	args=args_tr
	...
)
```

# `pipeline`
- Reference: https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html
```python
from transformers import pipeline

# `task`: (`"zero-shot-classification"`, `"”sentiment-analysis"`, `"ner"`, "`question-answering"`)
model = pipeline(task, model, [tokenizer])
```