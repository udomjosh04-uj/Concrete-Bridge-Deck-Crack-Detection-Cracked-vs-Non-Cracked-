# Streamlit file
import os
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Concrete Crack Screening", layout="centered")
st.title("Explainable Concrete Crack Screening Tool")
st.warning(
    "Research screening prototype. This tool does not determine structural safety, "
    "crack severity, or replace inspection by a qualified structural engineer."
)
