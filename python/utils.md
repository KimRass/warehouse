# Reload Module
```python
[import <module>]

from importlib import reload
import sys
reload(sys.modules["<module>"])
...
```

# `bs4`
```python
from bs4 import BeautifulSoup as bs

soup = bs(xml,"lxml")

soup.find_all()
soup.find_all().find()
soup.find_all().find().get_text()
```

# `selenium`
```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains

driver = webdriver.Chrome("chromedriver.exe")
driver.get("https://www.google.co.kr/maps/")
```

# `shutil`
```python
import shutil
```
## Copy File
```python
shutil.copy(src, dst)
```
## `shutil.copyfileobj()`
```python
shutil.copyfileobj(urllib3.PoolManager().request("GET", url, preload_content=False), open(file_dir, "wb"))
```

# `IPython`
```python
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")
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

# Returns the system/OS name, such as `'Linux'`, `'Darwin'`, `'Java'`, `'Windows'`. An empty string is returned if the value cannot be determined.
if platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = mpl.font_manager.FontProperties(
		fname="C:/Windows/Fonts/malgun.ttf"
	).get_name()
    mpl.rc("font", family=font_name)
```

# `pprint`
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

tar = tarfile.open(name)
tar.extractall([path])
tar.close()
```

# `lxml`
```python
from lxml import etree

with zipfile.ZipFile("ted_en-20160408.zip", "r") as z:
	target_text = etree.parse(z.open("ted_en-20160408.xml", "r"))
```

# `os`
- Reference: https://docs.python.org/3/library/os.html
```python
import os
```
## `os.getcwd()`
## Create Directory
```python
# 하나의 폴더만 생성할 수 있습니다.
os.mkdir(path)
# Like `mkdir()`, but makes all intermediate-level directories needed to contain the leaf directory.
# `exist_ok`: If `False` (the default), an `FileExistsError` is raised if the target directory already exists.
os.makedirs(path, exist_ok=True)
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
```python
# Returns the directory name of pathname `path`.
os.path.dirname()
# Returns the base name of pathname `path`.
os.path.basename()
```
## Check If File or Directory
```python
# Check If File
os.path.isfile()
# Check If Directory
os.path.isdir()
```
## Remove File or Directory
```python
# Remove file
os.remove()
# Remove directory
os.rmdir()
```

# `pathlib`
- Reference: https://docs.python.org/3/library/pathlib.html
```python
from pathlib import Path

# The logical parent of the path.
Path("...").parent
# The final path component with its suffix.
Path("...").name
# The final path component without its suffix.
Path("...").stem
# The file extension of the final component.
Path("...").suffix

Path("...").iterdir()

Path("...").glob()

# This is like calling `Path.glob()` with `"**/"` added in front of the given relative pattern.
# `"**"`: This directory and all subdirectories, recursively.
Path("...").rglob()

Path("...").is_file()
Path("...").is_dir()

Path("...").exists()
```

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

# `glob`
```python
path = "./DATA/전체"
filenames = glob.glob(path + "/*.csv")
```

# `pickle`
- Reference: https://docs.python.org/3/library/json.html
```python
import pickle as pk

pk.dump()
pk.load()
```

# `json`
```python
import json

# Save
# For Dictionary
# `ensure_ascii`: If `True`, the output is guaranteed to have all incoming non-ASCII characters escaped. If `False`, these characters will be output as-is.
# `indent`: If a string (such as `"\t"`), that string is used to indent each level.
with open(..., mode="w") as f:
	json.dump(obj, f, [ensure_ascii=True], [indent])
# For JSON string
with open(..., mode="w") as f:
	f.write(json_str)

# Load
with open(..., mode="r") as f:
	file = json.load(f)
```

# Progress Bar
```python
# On terminal
from tqdm import tqdm
# On Jupyter
from tqdm.notebook import tqdm
# Automatically chooses beween `tqdm.notebook` and `tqdm.tqdm`.
from tqdm.auto import tqdm
```
 
# Warning Control
```python
import warnings

# `category`: (`DeprecationWarning`, `UserWarning`)
warnings.filterwarnings("ignore", [category])
```

# Download Files
- Using `urllib.request.urlretrieve()`
	```python
	from urllib.request import urlretrieve
	
	urlretrieve(url, filename)
	```
- Using `google_drive_downloader.GoogleDriveDownloader.download_file_from_google_drive()`
	```python
	# Install: `pip install googledrivedownloader`
	from google_drive_downloader import GoogleDriveDownloader as gdd

	gdd.download_file_from_google_drive(file_id, dest_path)
	```
- Using `tensorflow.keras.utils.get_file()`
	```python
	from tensorflow.keras.utils import get_file

	# `fname`: Name of the file. If an absolute path is specified the file will be saved at that location. If `None`, the name of the file at origin will be used. By default the file is downloaded to `~/.keras/datasets`.
	# `origin`: Original URL of the file.
	path_to_downloaded_file = get_file([fname], origin, [untar])
	```

# `openpyxl`
```python
import openpyxl

wb = openpyxl.Workbook()
wb = openpyxl.load_workbook("D:/디지털혁신팀/태블로/HR분석/FINAL/★직급별인원(5년)_본사현장(5년)-태블로적용.xlsx")

from openpyxl.worksheet.formula import ArrayFormula

f = ArrayFormula("E2:E11", "=SUM(C2:C11*D2:D11)")

ws = wb[]
ws = wb.active
ws = wb.worksheets[0]

wb.create_sheet("Index_sheet")

df = ws[]
df = pd.DataFrame(ws.values)

# Insert
ws.insert_rows()
ws.insert_cols()

# Delete
ws.delete_rows()
ws.delete_cols()

# Append
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
ws.append(content)

# Merge
ws.merge_cells("A2:D2")
# Unmerge
ws.unmerge_cells()

wb.sheetnames
wb.save("test.xlsx")
```

# `pptx`
```python
from pptx import Presentation

prs = Presentation("sample.pptx")

prs.slides[<<Index of the Slide>>].shapes[<<Index of the Text Box>>].text_frame.paragraphs[<<Index of the Text in the Text Box>>].text

prs.slides[a].shapes[b].text_frame.paragraphs[c].font.name = "Arial"

prs.slides[a].shapes[b].text_frame.paragraphs[c].font.size = Pt(16)

# Save
prs.save("파일 이름")
```

# `ray`
```sh
pip install -U ray
```
```python
import ray

ray.init(ignore_reinit_error=True)
# Or
ray.shutdown()
ray.init()
...

@ray.remote
def func(...):
	...
...

# `ray.put()`: 크기가 큰 변수에 대해 사용.
res = ray.get([func.remote(ray.put(...), ...) for _ in range(n)])
```

# Log
- Reference: http://oniondev.egloos.com/v/9603983
```python
import logging
from pathlib import Path
import datetime

class Logger():
	def __init__(self, out_dir, save_each):
		if not isinstance(out_dir, Path):
			out_dir = Path(out_dir)

		self.out_dir = out_dir
		self.save_each = save_each

	def get_logger(self):
		logger = logging.getLogger(__name)
		logger.propagate = False
		
		# Example
		# `levelname`, `message`, `asctime`, `filename`, `funcName`, `lineno`...
			# `asctime`: The numbers after the comma are millisecond portion of the time.
		formatter = logging.Formatter(
			fmt="| %(asctime)s | %(levelname)-8s | %(message)-66s | %(filename)-22s | %(funcName)-32s |",
			datefmt="%Y-%m-%d %H:%M:%S"
		)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)
		
		# (`logging.DEBUG`, `logging.INFO`, `logging.WARNING`, `logging.ERROR`, `logging.CRITICAL`)
		logger.setLevel(logging.DEBUG)

		if self.save_each:
			file_handler = logging.FileHandler(
				self.out_dir / f"errors_{datetime.now().strftime('%Y-%m-%d% %H:%M:%S')}.log"
			)
		else:
			file_handler = logging.FileHandler(
				self.out_dir / "errors.log"
			)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		return logger
		...

logger = Logger(...).get_logger()
...
```

# Parse Arguments
- Reference: https://docs.python.org/3/library/argparse.html
```python
import argparse

# Example
def get_args():
	# `"description"`: Text to display before the argument help.
    parser = argparse.ArgumentParser([description=None])

	# `action`
		# `action="store"`: (default) This just stores the argument’s value.
		# `action="store_const"': This stores the value specified by the `const` keyword argument. The `store_const` action is most commonly used with optional arguments that specify some sort of flag.
		# `action="store_true"`: Stores the value `True`.
		# `action="store_false"`: Stores the value `False`.
	# `default`: Specifies what value should be used if the command-line argument is not present.
	# `type`: (default `str`) The type to which the command-line argument should be converted.
	# `nargs`
		# `nargs=N`: `N` arguments from the command line will be gathered together into a List.
		# 'nargs="*"'. All command-line arguments present are gathered into a List.
	# `help`: A brief description of what the argument does.
	# `dest`: The name of the attribute to be added to the object returned by `parse_args()`.
		# The value of `dest` is normally inferred from the option strings. `ArgumentParser` generates the value of `dest` by taking the first long option string and stripping away the initial `"--"` string. If no long option strings were supplied, `dest` will be derived from the first short option string by stripping the initial `"-"` character.
	# Example
	parser.add_argument("-o", "--out_path", ..., dest="out_path", action="store")
	...

    return parser.parse_args()

...
args = get_args()
out_path = args.out_path
...
```

# Read Configuration
```python
from pathlib import Path
import json

def read_config():
    config_json = Path(__file__).parent / "config.json"
    with open(config_json, mode="r") as f:
        config = json.load(f)
    ... = config["..."]
	...
    return ...
...

... = read_config()
```

# Run Shell Script in Python
```python
import subprocess

# Examples
subprocess.run(f"bash list_files.sh s3://{bucket} {s3_dir} {local_dir}", shell=True)
subprocess.run(
	f"aws s3 cp {local_dir}{i} {bucket}/{s3_dir} --acl bucket-owner-full-control",
	shell=True,
)
```

# `pyinstaller`
```sh
# `option`:
	# `--onefile`: One file
	# `--onedifr`: One folder
pyinstaller [option] <file_name>.py
```

# Validate JSON Schema
```python
import json
from jsonschema import validate

with open("/path/to/json_schema.json", mode="r") as f:
    json_schema = json.load(f)
    
with open("/path/to/json_to_validate.json", mode="r") as f:
    json_to_validate = json.load(f)
    
# `"enum"
	# The value of this keyword MUST be an array. This array SHOULD have at least one element. Elements in the array SHOULD be unique. An instance validates successfully against this keyword if its value is equal to one of the elements in this keyword's array value. Elements in the array might be of any type, including null.
# `pattern`
	# The value of this keyword MUST be a string. This string SHOULD be a valid regular expression.
validate(schema=json_schema, instance=json_to_validate)
```

```sh
pip install google-auth-oauthlib
pip install google-api-python-client
pip install getfilelistpy
```

# `typing`
```python
from typing import Any, Dict, List, Optional, Union, Tuple
...

def ...(...) -> Tuple[list, ...]:
	...
```