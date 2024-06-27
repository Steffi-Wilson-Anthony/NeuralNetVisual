
import streamlit as st
from collections.abc import Iterable
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = 'http://127.0.0.1:5000'

st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input Image')

if st.button('Get random prediction'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))
    
    st.sidebar.image(image, width=150)
    
    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        fig = plt.figure(figsize=(32,4))
        
        if layer==2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16
        for i, number in enumerate(numbers):
            ax = plt.subplot(row, col, i+1)
            ax.imshow(number * np.ones((8, 8, 3)).astype('float32'))
            ax.set_xticks([])
            ax.set_yticks([])
            
            if layer==2:
                ax.set_xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text('Layer {}'.format(layer+1))
        st.pyplot(fig)
