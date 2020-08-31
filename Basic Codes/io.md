# io
## BytesIO
```python
from io import BytesIO
```
```python
url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/img/POSTER_20M_FULL/{}.jpg".format(movie_id)
req = requests.get(url)
b = BytesIO(req.content)
img = np.asarray(Image.open(b))
```
