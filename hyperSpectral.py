import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np


st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')


if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    # cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                             # key = 2)
    # cropped_img2= st_cropper(img, realtime_update=realtime_update, box_color=box_color,
    #                          aspect_ratio=aspect_ratio, key=1)
    # Manipulate cropped image at will

    # _ = cropped_img.thumbnail((150, 150))
    # st.image(cropped_img)
    # st.image(cropped_im
    # figg2)

    im = np.array(Image.open(img_file))
    fig = go.Figure(data=go.Heatmap(
        z=im))
    st.plotly_chart(fig)
