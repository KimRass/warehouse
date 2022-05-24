# `bs4`
```python
from bs4 import BeautifulSoup as bs

soup = bs(xml,"lxml")

soup.find_all()
soup.find_all().find()
soup.find_all().find().get_text()
```

# `selenium`
## `webdriver`
```python
from selenium import webdriver
```
```python
driver = webdriver.Chrome("chromedriver.exe")
```
### `driver.get()`
```python
driver.get("https://www.google.co.kr/maps/")
```
### `driver.find_element_by_css_selector()`, `driver.find_element_by_tag_name()`, `driver.find_element_by_class_name()`, `driver.find_element_by_id()`, `driver.find_element_by_xpath()`,
#### `driver.find_element_by_*().text`
```python
df.loc[index, "배정초"]=driver.find_element_by_xpath("//\*[@id='detailContents5']/div/div[1]/div[1]/h5").text
```
#### `driver.find_element_by_*().get_attribute()`
```python
driver.find_element_by_xpath("//*[@id='detailTab" +str(j) + "']").get_attribute("text")
```
#### `driver.find_element_by_*().click()`
#### `driver.find_element_by_*().clear()`
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').clear()
```
#### `driver.find_element_by_*().send_keys()`
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').send_keys(qeury)
```
```python
driver.find_element_by_name('username').send_keys(id)
driver.find_element_by_name('password').send_keys(pw)
```
```python
driver.find_element_by_xpath('//*[@id="wpPassword1"]').send_keys(Keys.ENTER)
```
### `driver.execute_script()`
```python
for j in [4,3,2]:
    button = driver.find_element_by_xpath("//\*[@id='detailTab"+str(j)+"']")
    driver.execute_script("arguments[0].click();", button)
```
### `driver.implicitly_wait()`
```python
driver.implicitly_wait(1)
```
### `driver.current_url`
### `driver.save_screenshot()`
```python
driver.save_screenshot(screenshot_title)
```
## `WebDriverWait()`
### `WebDriverWait().until()`
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
```
```python
WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, "//\*[@id='detailContents5']/div/div[1]/div[1]/h5")))
```
- `By.ID`, `By.XPATH`
## `ActionChains()`
```python
from selenium.webdriver import ActionChains
```
```python
module=["MDM","사업비","공사","외주","자재","노무","경비"]

for j in module:
    module_click=driver.find_element_by_xpath("//div[text()='"+str(j)+"']")
    actions=ActionChains(driver)
    actions.click(module_click)
    actions.perform()
```
### `actions.click()`, `actions.double_click()`

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
# `logging`
## `logging.basicConfig()`
```python
logging.basicConfig(level=logging.ERROR)
```
# `IPython`
## `IPython.display`
### `set_matplotlib_formats`
```python
from IPython.display import set_matplotlib_formats
```
```python
set_matplotlib_formats("retina")
```
- font를 선명하게 표시
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

path = "C:/Windows/Fonts/malgun.ttf"
# Returns the system/OS name, such as `'Linux'`, `'Darwin'`, `'Java'`, `'Windows'`. An empty string is returned if the value cannot be determined.
if platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    font_name = mpl.font_manager.FontProperties(fname=path).get_name()
    mpl.rc('font', family=font_name)
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

tar = tarfile.open("buzzni.tar")
tar.extractall(path)
tar.close()
```

# `lxml`
```python
from lxml import etree

with zipfile.ZipFile("ted_en-20160408.zip", "r") as z:
	target_text = etree.parse(z.open("ted_en-20160408.xml", "r"))
```

# `os`
```python
import os
```
## `os.getcwd()`
## Create Directory
```python
# 하나의 폴더만 생성할 수 있습니다.
os.mkdir(path)
# 여러 개의 폴더를 생성할 수 있습니다.
os.makedirs(path, exist_ok=True)
```
## `os.chdir()`
## `os.environ`
## `os.pathsep`
```python
os.environ["PATH"] += os.pathsep + "C:\Program Files (x86)/Graphviz2.38/bin/"
```
## `os.path`
### `os.path.join()	`
```python
os.path.join("C:\Tmp", "a", "b")
```
### `os.path.exists()`
```python
if os.path.exists("C:/Users/5CG7092POZ/train_data.json"):
```
```python
# Returns the directory name of pathname `path`.
os.path.dirname()
# Returns the base name of pathname `path`.
os.path.basename()
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
# `ensure_ascii`: If `True` (the default), the output is guaranteed to have all incoming non-ASCII characters escaped. If `False`, these characters will be output as-is.
# `indent`: If a string (such as `"\t"`), that string is used to indent each level.
json.dump(obj, fp, [ensure_ascii], [indent])

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

# `category`: (`DeprecatingWarning`)
warnings.filterwarnings("ignore", [category])
```

# Download Files
- Using `urllib.request.urlretrieve()`
	```python
	import urllib.request
	
	urllib.request.urlretrieve(url, filename)
	```
- Using `google_drive_downloader.GoogleDriveDownloader.download_file_from_google_drive()`
	```python
	# Install: `pip install googledrivedownloader`
	
	from google_drive_downloader import GoogleDriveDownloader as gdd
	gdd.download_file_from_google_drive(file_id, dest_path)
	```
- Using `tensorflow.keras.utils.get_file()`
	```python
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
pip install ray
```
```python
import ray

ray.init(ignore_reinit_error=True)
...

@ray.remote
def func(...):
	...
...

# `ray.put()`: 크기가 큰 변수에 대해 사용.
res = ray.get([func.remote(ray.put(...), ...) for _ in range(n)])
```

# `logging`
```python
import logging

# logging의 기본 level 변경
# `format`
	# `levelname`
	# `message`
	# `asctime`
	# `filename`
	# `funcName`
	# `lineno`
# `datefmt`
logging.basicConfig([filename], [format], [datefmt], level=logging.DEBUG)

logging.debug("디버그")
logging.info("정보")
logging.warning("경고")
logging.error("에러")
logging.critical("심각")

# 예외 처리
try:
	...
except Exception:
	loggng.exception()
```
```python
logger = logging.getLogger(__name__)
# `getLogger()`를 이용하여 가져온 logger가 이미 핸들러를 가지고 있다면, 이미 핸들러가 등록되어 있으므로 새로운 핸들러를 또 등록할 이유가 없다. (Source: https://5kyc1ad.tistory.com/269)
# `addHandler()` does not check if a similar handler has already been added to the logger. (Source: https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging)
logger.propagate = False

formatter = logging.Formatter('%(name)s - %(message)s')

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.addHandler(handler)
...
```

# `pyinstaller`
```sh
# `option`:
	# `--onefile`: One file
	# `--onedifr`: One folder
pyinstaller [option] <file_name>.py
```
