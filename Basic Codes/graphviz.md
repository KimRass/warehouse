# graphviz
```python
import graphviz
```
```python
def plot_tree(model, filename, rankdir="UT"):
    import os
    gviz = xgb.to_graphviz(model, num_trees = model.best_iteration, rankdir=rankdir)
    _, file_extension = os.path.splitext(filename)
    format = file_extension.strip(".").lower()
    data = gviz.pipe(format=format)
    full_filename = filename
    with open(full_filename, "wb") as f:
        f.write(data)
```
