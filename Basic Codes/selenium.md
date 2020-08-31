# selenium
## webdriver
```python
from selenium import webdriver
```
```python
driver = webdriver.Chrome("chromedriver.exe")
```
### driver.get()
```python
driver.get("https://www.google.co.kr/maps/")
```
### driver.find_element_by_css_selector(), driver.find_element_by_tag_name(), driver.find_element_by_class_name(), driver.find_element_by_id(), driver.find_element_by_xpath(),
#### driver.find_element_by_*().text
```python
df.loc[index, "배정초"]=driver.find_element_by_xpath("//\*[@id='detailContents5']/div/div[1]/div[1]/h5").text
```
#### driver.find_element_by_*().get_attribute()
```python
driver.find_element_by_xpath("//*[@id='detailTab" +str(j) + "']").get_attribute("text")
```
#### driver.find_element_by_*().click()
#### driver.find_element_by_\*().clear()
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').clear()
```
#### driver.find_element_by_\*().send_keys()
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
### driver.execute_script()
```python
for j in [4,3,2]:
    button = driver.find_element_by_xpath("//\*[@id='detailTab"+str(j)+"']")
    driver.execute_script("arguments[0].click();", button)
```
### driver.implicitly_wait()
```python
driver.implicitly_wait(1)
```
### driver.current_url
### driver.save_screenshot()
```python
driver.save_screenshot(screenshot_title)
```
## WebDriverWait(), By, EC
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
```
```python
WebDriverWait(driver, wait_sec).until(EC.presence_of_element_located((By.XPATH,"//\*[@id='detailContents5']/div/div[1]/div[1]/h5")))
```
## ActionChains()
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
### actions.click(), actions.double_click()
