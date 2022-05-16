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
## `shutil.copyfile()`
```python
shutil.copyfile("./test1/test1.txt", "./test2.txt")
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
## `os.path`
### `os.path.join()	`
```python
os.path.join("C:\Tmp", "a", "b")
```
### `os.path.exists()`
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

# `pathlib`
- Reference: https://docs.python.org/3/library/pathlib.html
```python
from pathlib import Path

# The final path component, without its suffix.
Path("...").stem
```

# `glob`
## `glob.glob()`
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

# `tqdm`
- For VSCode
	```python
	from tqdm import tqdm
	```
- For Jupyter Notebook
	```python
	from tqdm.notebook import tqdm
	```
- For Google Colab
	```python
	from tqdm.auto import tqdm
	```
 
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

# `openpyxl`
```python
import openpyxl
```
## `openpyxl.Workbook()`
```python
wb = openpyxl.Workbook()
```
## `openpyxl.load_workbook()`
```python
wb = openpyxl.load_workbook("D:/디지털혁신팀/태블로/HR분석/FINAL/★직급별인원(5년)_본사현장(5년)-태블로적용.xlsx")
```
## `openpyxl.worksheet`
### `openpyxl.worksheet.formula`
#### `ArrayFormula`
```python
from openpyxl.worksheet.formula import ArrayFormula
```
```python
f = ArrayFormula("E2:E11", "=SUM(C2:C11*D2:D11)")
```
### `wb[]`
### `wb.active`
```python
ws = wb.active
```
### `wb.worksheets`
```python
ws = wb.worksheets[0]
```
### `wb.sheetnames`
### `wb.create_sheet()`
```python
wb.create_sheet("Index_sheet")
```
#### `ws[]`
#### `ws.values()`
```python
df = pd.DataFrame(ws.values)
```
#### `ws.insert_rows()`, `ws.insert_cols()`, `ws.delete_rows()`, `ws.delete_cols()`
#### `ws.append()`
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
ws.append(content)
```
#### `ws.merge_cells()`, `ws.unmerge_cells()`
```python
ws.merge_cells("A2:D2")
```
### `wb.sheetnames`
### `wb.save()`
```python
wb.save("test.xlsx")
```
### `wb.active`
```python
sheet = wb.active
```
### `wb.save()`
```python
wb.save("test.xlsx")
```
#### `sheet.append()`
```python
content = ["민수", "준공분", "거제2차", "15.06", "18.05", "1279"]
sheet.append(content)
```
#### `sheet[]`
```python
sheet["H8"] = "=SUM(H6:H7)"
```

# `pptx`
## `Presentation()`
```python
from pptx import Presentation
```
```python
prs = Presentation("sample.pptx")
```
### `prs.slides[].shapes[].text_frame.paragraphs[].text`
```python
prs.slides[<<Index of the Slide>>].shapes[<<Index of the Text Box>>].text_frame.paragraphs[<<Index of the Text in the Text Box>>].text
```
### `prs.slides[].shapes[].text_frame.paragraphs[].font`
#### `prs.slides[].shapes[].text_frame.paragraphs[].text.name`
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.name = "Arial"
```
#### `prs.slides[].shapes[].text_frame.paragraphs[].text.size`
```python
prs.slides[a].shapes[b].text_frame.paragraphs[c].font.size = Pt(16)
```
### `prs.save()`
```python
prs.save("파일 이름")
```