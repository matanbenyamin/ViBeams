import xgboost as xgb
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import cv2
import plotly.express as px
import pandas as pd
from PIL import Image,ImageOps
import streamlit as st


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i



def power(image, kernel, normalized = True):
    # Normalize images for better comparison.
    if normalized:
      image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)



reg = xgb.XGBRegressor()
reg.load_model('model.bin')

cons = ['10','5','2_5','1_25', '063','0']
con = cons[np.random.randint(len(cons))]
path = '/content/drive/MyDrive/Lab_Experiments/17Feb2022 - Michael Lab/' + con + 'M x10 6Well-Plate/'
print(path)
filename = str(np.random.randint(8)).zfill(4) + '.bmp'

img_file = st.sidebar.file_uploader(label='Upload a file', type=['bmp'])
im = np.array(ImageOps.grayscale(Image.open(img_file)))


results = []
kernel_params = []
dft = pd.DataFrame()
for theta in (0,1):
    theta = theta / 4. * np.pi
    for frequency in np.arange(0.5, 1.5, 0.1):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append( np.mean(power(im, kernel)))
dft = dft.append(pd.DataFrame([results], index = [0]))
dft.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19']


fig = px.imshow(im)
st.plotly_chart(fig)
st.write(str(reg.predict(dft)))
