import streamlit as st

st.title("st.title")
st.header("st.header")
st.subheader("st.subheader")
st.text("st.text")
st.markdown("# `sort_index()`\n"
    "## `set_index([drop])`\n"
    "### `reset_index([drop], [level])`")
    
st.success("st.success")
st.info("st.info")
st.warning("st.warning")
st.error("st.error")
st.exception("st.exception")

from PIL import Image

img = Image.open("D:/디지털혁신팀/태블로/홈 화면/menu design 샘플/img.png")
st.image(img, width=1000, caption="caption")

with open(file="D:/Github/Work/Reverse Mentoring/shake dat booty (Wahoo Main Mix)_babis.mp3", mode="rb") as f:
    aud = f.read()
st.audio(aud, start_time=30)

st.checkbox("st.checkbox1")
st.checkbox("st.checkbox2")
st.checkbox("st.checkbox3")

# Single Value Box?
status = st.radio("Message", ("Name1", "Name2", ...))
if status == "Name1":
    st.write("st.write1")
elif status == "Name2":
    st.write("st.write2")
    
status = st.radio("st.radio", ("Button1", "Button2"))
if status == "Button1":
    st.write("st.write1")
else:
    st.write("st.write2")
    
case = st.selectbox("Message", ["Case1", "Case2", "Case3"])
st.write(case)

case = st.multiselect("...", ["Case1", "Case2", ...])
st.write(len(case), case)

age = st.slider('How old are you?', 0, 130, (23, 44))
st.write("I'm ", age, 'years old')

from datetime import datetime
start_time = st.slider(
     "When do you start?",
     value=datetime(2020, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)

     
import streamlit as st

def get_user_name():
    return 'John'

with st.echo(code_location="below"):
    st.write("a")
st.write("b")

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(10, 20), columns=('col %d' % i for i in range(20)))

st.dataframe(df)


with st.sidebar:
    st.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)
import time

e = ValueError('...')
st.exception(e)