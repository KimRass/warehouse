- Reference: https://docs.streamlit.io/library/api-reference?highlight=html#display-progress-and-status
```python
import streamlit as st
```
# Run
```python
streamlit run ....py
```
# Text Elements
```python
st.title("...")
st.header("...")
st.subheader("...")
st.text("...")
st.markdown("...")
st.latex(r"...")
```
# Code Block
## For One-lined Code
```python
st.code(body, language="python")
```
## Code Snippet
- Sometimes you want your Streamlit app to contain both your usual Streamlit graphic elements and the code that generated those elements. That's where st.echo() comes in.
```python
# `code_location`: (`"above"`, `"below"`, default `"above"`) Whether to show the echoed code before or after the results of the executed code block.
with st.echo([code_location]):
	# Everything inside this block will be both printed to the screen and executed.
    ...
```
# Error/Colorful Text
```python
st.success("...")
st.info("...")
st.warning("...")
st.error("...")
st.exception("...")
```
# DataFrame
```python
st.dataframe(data, [width], [height])
```
# Spinner
- Temporarily displays a message while executing a block of code.
```python
with st.spinner(text):
	...
...
```
# Media Elements
## Show Image
```python
from PIL import Image

img = Image.open(fp)
st.image(img, [width], [caption])
```
## Show Video
```python
with open(file, mode="rb") as f:
    vid = f.read()
st.video(vid, [start_time)
```
## Insert Audio
```python
with open(file, mode="rb") as f:
    aud = f.read()
st.audio(aud, [start_time])
```
# Widgets
## Button
```python
clicked = st.button()
```
## Download Button
```python
st.download_button()
```
## Checkbox
```python
selected = st.checkbox("...")
```
## Radio Button
```python
status = st.radio("...", ("Button1", "Button2"))
if status == "Button1":
    st.write("...")
else:
    st.write("...")
```
## Dropdown
```python
case = st.selectbox("...", ["Case1", "Case2", ...])
st.write(case)
```
## Multi-Selection Dropdown
```python
case = st.multiselect("...", ["Case1", "Case2", ...])
st.write(case)
```
## Slider
```python
# `label`: A short label explaining to the user what this slider is for.
# `value`: The value of the slider when it first renders. If a Tuple/List of two values is passed here, then a range slider with those lower and upper bounds is rendered.
# `step`: The stepping interval. Defaults to `1` if the value is an Int, `0.01` if a Float, `timedelta(days=1)` if a Date/Datetime, `timedelta(minutes=15)` if a Time.
pick = st.slider(label, min_value, max_value, value, step)
```
## Text Input
```python
text = st.text_input()
```
## Number Input
```python
num = st.number_input()
```
## Date Input
```python
data = st.date_input()
```
## Time Input
```python
time = st.time_input()
```
## Camera Input
```python
img = st.camera_input()
```
## Color Picker
```python
color = st.color_picker()
```
## Upload File
```python
file = st.file_uploader()
```
# Progress Bar
```python
# Create progress bar
bar = st.progress(0)
for ... in ...:
	# `value`: 0 <= `value` <= 100 for Int. 0.0 <= `value` <= 1.0 for Float.
	bar.progress(value)
```