# json
## json.load(), json.dump()
```python
path = "C:/Users/5CG7092POZ/train_data.json"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8"):
        train_data = json.load(f)
else:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent="\t")
```
