import cv2
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
import numpy as np
from PIL import Image



col1, col2, col3, col4 = st.columns((2,1,1,1))

uploaded_file =  st.sidebar.file_uploader(label='Upload a file', type=['bmp', 'png', 'jpg'])
if uploaded_file is not None:
    im = np.array(Image.open(uploaded_file))
    # path = "C:\\Users\lab7\Downloads\ ".strip()
    # im_name = 'cells1.bmp'
    # im = cv2.imread(path + im_name)
    im = im[0::5,0::5]

    with col1:
        fig = px.imshow(im,  color_continuous_scale='gray')
    # fig.data[0].update(mode = 'marker')
    # st.plotly_chart(fig)


    selected_points = plotly_events(fig)
    selected_points
    a=selected_points[0]
    fig.add_scatter(x=[a["x"]], y=[a["y"]], fillcolor='red')



    # streamlit run myApp.py

    app_state = st.experimental_get_query_params()

    # Display saved result if it exist
    if "prev_point" in app_state:
        prev_point = app_state["prev_point"]
        fig.add_scatter(x=[prev_point[0]], y=[prev_point[1]],fillcolor = 'rgb(0,1,1)')
        selected_points = plotly_events(fig)



    if "state" in app_state:
        curr_state = int(app_state["state"][0])
        curr_state = -curr_state
    else:
        curr_state = 1



    # Perform an operation, and save its result
    st.experimental_set_query_params(prev_point=[a["x"], a["y"]], state = curr_state)  # Save value

    if "prev_point" in app_state:
        spectrum = im[a["y"]:a["y"]+5,a["x"]:a["x"]+5]
        spectrum = spectrum.flatten()


        int(prev_point[1])

        spectrum_p = im[int(prev_point[1]):int(prev_point[1]) + 5, int(prev_point[0]):int(prev_point[0]) + 5]
        spectrum_p = spectrum_p.flatten()

        freqs = [
        910,919,930,639,949,
        850,862,877,889,897,
        786,801,812,825,840,
        726,738,751,766,779,
        656,667,685,697,710]
        df = pd.DataFrame(index = freqs, data = [[spectrum]],columns=['current'])
        df = df.sort_index()

        df_p = pd.DataFrame(index=freqs, data=[[spectrum_p]], columns=['previous'])
        df_p = df_p.sort_index()

        df = df.join(df_p)

        # plot
        with col2:
            fig2 = px.line(df, title='Spectrum')
        curr_state
        if curr_state>0:
            fig2.data[0].line.color = "red"
            fig2.data[1].line.color = "blue"
        else:
            fig2.data[1].line.color = "red"
            fig2.data[0].line.color = "blue"

        # fig2.add_scatter(x = df_p.index, y = df_p['1'])


        st.plotly_chart(fig2)
        # streamlit run myApp.py

