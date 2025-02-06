import streamlit as st

@st.cache_data
def expensive_computation(x):
    return x * x

# Clear Streamlit's cache
st.cache_data.clear()
