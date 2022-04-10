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