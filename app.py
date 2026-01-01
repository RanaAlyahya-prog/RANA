import streamlit as st
import pandas as pd

st.title("NYC Crime Dashboard")
st.write("Welcome to the NYC Crime Analysis Dashboard")

data = {"Weapon": ["Gun", "Knife", "Other"], "Total Homicides": [120, 50, 30]}
df = pd.DataFrame(data)

st.subheader("Homicides by Weapon")
st.dataframe(df)
st.bar_chart(df.set_index("Weapon"))
