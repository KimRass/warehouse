# `random`, `np.random`, `tf.random`
## Seed
### `random.seed()`
### `np.random.seed()`
### `tf.random.set_seed()`
### `tf.random.normal()`
## Sample
### `random.random()`
- Returns a random number in [0, 1).
### `np.random.rand()`
- Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
### `random.sample(sequence, k)`
### `random.choices(sequence, k, [weights])`
### `np.random.choice(size, [replace], [p])`
- Generates a random sample from a given 1-D array.
- `replace`: (bool)
- `p`: The probabilities associated with each entry in `a`. If not given, the sample assumes a uniform distribution over all entries in `a`
### `random.randint(a, b)`
- Return a random integer N such that `a` <= N <= `b`.
- Alias for `randrange(a, b+1)`.
### `np.random.randint(low, [high], size)`
- Return random integers from `low` (inclusive) to `high` (exclusive).
- Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval [`low`, `high`). If `high` is None (the default), then results are from [`0`, `low`).
## Shuffle
### `random.shuffle()`
- In-place function

# `copy`
## `copy.copy()`
## `copy.deepcopy()`

# `shutil`
```python
import shutil
```
## `shutil.copyfile()`
```python
shutil.copyfile("./test1/test1.txt", "./test2.txt")
```
## `shutil.copyfileobj()`
```python
shutil.copyfileobj(urllib3.PoolManager().request("GET", url, preload_content=False), open(file_dir, "wb"))
```

# `itertools`
```python
from itertools import combinations, permutations, product, combinations_with_replacement
```
```python
movies = {a | b for a, b in combinations(movie2sup.keys(), 2)}
```
```python
# `repeat`
for i in product(range(3), range(3), range(3)):
    print(i)
```

# `collections`
## `Counter()`
```python
from collections import Counter
```
```python
word2cnt = Counter(words)
```
- lst의 원소별 빈도를 나타내는 dic을 반환합니다.
### `Counter().values()`
```python
sum(Counter(nltk.ngrams(cand.split(), 2)).values())
```
### `Counter().most_common()`
## `deque()`
```python
from collections import deque
```
```python
dq = deque("abc")
```
- `maxlen`
### `dq.append()`
### `dq.appendleft()`
### `dq.pop()`
### `dq.popleft()`
### `dq.extend()`
### `dq.extendleft()`
### `dq.remove()`
## `defaultdict()`
```python
ddic = defaultdict(list)
```

# `functools`
## `reduce()`
```python
from functools import reduce
```
```python
reduce(lambda acc, cur: acc + cur["age"], users, 0)
```
```python
reduce(lambda acc, cur: acc + [cur["mail"]], users, [])
```

# `platform`
```python
import platform
```
## `platform.system()`
```python
path = "C:/Windows/Fonts/malgun.ttf"
if platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = mpl.font_manager.FontProperties(fname=path).get_name()
    mpl.rc('font', family=font_name)
```
- Returns the system/OS name, such as `'Linux'`, `'Darwin'`, `'Java'`, `'Windows'`. An empty string is returned if the value cannot be determined.

# `pprint`
## `pprint()`
```python
from pprint import pprint
```

# `zipfile`
```python
import zipfile

zip = zipfile.ZipFile(file, mode)
zip.extractall(path)`
zip.close()
```

# `tarfile`
```python
import tarfile

tar = tarfile.open("buzzni.tar")
tar.extractall(path)
tar.close()
```

# `lxml`
## `etree`
```python
from lxml import etree
```
### `etree.parse()`
```python
with zipfile.ZipFile("ted_en-20160408.zip", "r") as z:
	target_text = etree.parse(z.open("ted_en-20160408.xml", "r"))
```

# `glob`
## `glob.glob()`
```python
path = "./DATA/전체"
filenames = glob.glob(path + "/*.csv")
```

# `pickle`
```python
import pickle as pk
```
## `pk.dump()`
## `pk.load()`

# `json`
- Reference: https://docs.python.org/3/library/json.html
```python
import json
```
## `json.dump(obj, fp, [ensure_ascii], [indent])`
- `ensure_ascii`: If `True` (the default), the output is guaranteed to have all incoming non-ASCII characters escaped. If `False`, these characters will be output as-is.
- `indent`: If a string (such as `"\t"`), that string is used to indent each level.
## `json.load(f)`

# `datasketch`
## `MinHash`
```python
from datasketch import MinHash
```
```python
mh = MinHash(num_perm=128)
```
- MinHash는 각 원소 별로 signature를 구한 후, 각 Signature 중 가장 작은 값을 저장하는 방식입니다. 가장 작은 값을 저장한다 해서 MinHash라고 불립니다.
### `mh.update()`
```python
for value in set_A:
    mh.update(value.encode("utf-8"))
```
### `mh.hashvalues`

# `redis`
## `Redis`
```python
from redis import Redis
```
```python
rd = Redis(host="localhost", port=6379, db=0)
```
### `rd.set()`
```python
rd.set("A", 1)
```
### `rd.delete()`
```python
rd.delete("A")
```
### `rd.get()`
```python
rd.get("A")
```

# `os`
```python
import os
```
## `os.getcwd()`
## `os.makedirs()`
```python
os.makedirs(ckpt_dir, exist_ok=True)
```
## `os.chdir()`
## `os.environ`
## `os.pathsep`
```python
os.environ["PATH"] += os.pathsep + "C:\Program Files (x86)/Graphviz2.38/bin/"
```
## `os.path.join()	`
```python
os.path.join("C:\Tmp", "a", "b")
```
## `os.path.exists()`
```python
if os.path.exists("C:/Users/5CG7092POZ/train_data.json"):
```
## `os.path.dirname()`
```python
os.path.dirname("C:/Python35/Scripts/pip.exe")
# >>> 'C:/Python35/Scripts'
```
- 경로 중 디렉토리명만 얻습니다.

# `sys`
```python
import sys
```
## `sys.maxsize()`
## `sys.path`
- Source: https://www.geeksforgeeks.org/sys-path-in-python/
- `sys.path` is a built-in variable within the sys module. It contains a list of directories that the interpreter will search in for the required module. When a module(a module is a python file) is imported within a Python file, the interpreter first searches for the specified module among its built-in modules. If not found it looks through the list of directories(a directory is a folder that contains related modules) defined by `sys.path`.
```python
sys.path.append("...")
```
## `sys.stdin`
### `sys.stdin.readline()`
```python
cmd = sys.stdin.readline().rstrip()
```
## `sys.getrecursionlimit()`
## `sys.setrecursionlimit()`
```python
sys.setrecursionlimit(10**9)
```

# `tqdm`
- For Jupyter Notebook
	```python
	from tqdm.notebook import tqdm
	```
- For Google Colab
	```python
	from tqdm.auto import tqdm
	```
## `tqdm.pandas()`
- `DataFrame.progress_apply()`를 사용하기 위해 필요합니다.

# `warnings`
```python
import warnings

# `category`: (`DeprecatingWarning`)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

# Download Files
- Using `urllib.request.urlretrieve()`
	```python
	import urllib.request
	
	urllib.request.urlretrieve(url, filename)
	```
- Using `google_drive_downloader.GoogleDriveDownloader.download_file_from_google_drive()`
	```python
	# Install: !pip install googledrivedownloader
	
	from google_drive_downloader import GoogleDriveDownloader as gdd

	gdd.download_file_from_google_drive(file_id, dest_path)
	```
- Using `tensorflow.keras.utils.get_file()`
	```python
	# `fname`: Name of the file. If an absolute path is specified the file will be saved at that location. If `None`, the name of the file at origin will be used. By default the file is downloaded to `~/.keras/datasets`.
	# `origin`: Original URL of the file.
	path_to_downloaded_file = get_file([fname], origin, [untar])
	```
