```python
!pip install --upgrade category_encoders
```
# category_encoders
```python
import category_encoders as ce
```
## ce.target_encoder
### ce.target_encoder.TargetEncoder()
```python
encoder = ce.target_encoder.TargetEncoder(cols=["company1"])
encoder.fit(data["company1"], data["money"]);
data["company1_label"] = encoder.transform(data["company1"]).round(0)
```
