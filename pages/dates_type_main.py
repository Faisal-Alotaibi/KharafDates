import pandas as pd
import numpy as np
from dash import html, dcc, register_page, callback
from dash import dcc, html, Output, Input, State
from dash.exceptions import PreventUpdate
from helpers import parse_content

import io
import pickle
from PIL import Image
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import base64

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split 

from sklearn.metrics import \
     classification_report, confusion_matrix,\
     accuracy_score, precision_score, recall_score, f1_score,roc_auc_score


import keras
from keras.models import Sequential , load_model
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


register_page(__name__, path='/',
              title='Dates Type', name='Dates Type')

# Dictionary mapping indices to class labels
class_mapping = {
    0: 'Ajwa',
    1: 'Mabroom',
    2: 'Sukkary'
}


layout = html.Div([
    html.Div([
        html.Div(
            [html.H6("Upload Image", className="title")]
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            # Do not Allow multiple files to be uploaded
            multiple=False,
            className='image-upload'
        ),
        html.Div(id='output-filename'),
    ]),
    html.Div([
        html.Button('Submit', id='submit-file-btn', n_clicks=0),
    ], className="process-btn-div"),
    html.Div(id='model_output', className="model-output-div"),
], className="twelve columns page_div")


# ------------- Callback Methods ----------------
@callback(Output('output-filename', 'children'),
          Input('upload-data', 'contents'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'),
          prevent_initial_call=True
          )
def render_image_preview(image, filename, date):
    if image is not None:
        children = [parse_content(image, filename, date)]
        return children
    else:
        raise PreventUpdate


model = load_model(r'model2.h5')  # Load the saved model
#model = load_model(r'C:\Users\FzoOT\Downloads\Faisalaletaibi_Dash_Dashboard\model2.h5')  # Load the saved model

def preprocess_image(image):
    # Decode base64 content and convert it to bytes
    decoded = base64.b64decode(image.split(",")[-1])  # Extract base64 content from data URL

    # Open the image using PIL
    img = Image.open(io.BytesIO(decoded))

    # Resize and preprocess the image
    img = img.resize((256, 256))  # Resize the image to desired dimensions
    img = np.array(img)  # Convert to NumPy array



    # Reshape the image if needed for model input
    img = img.reshape(256, 256, 3)

    return img

def predict_image(image):
    processed_image = preprocess_image(image)
    # Assuming the model.predict method expects a NumPy array as input
    prediction = model.predict(np.array([processed_image]))
    return prediction

@callback([Output('model_output', 'children'),
           ],
          Input('submit-file-btn', 'n_clicks'),
          State('upload-data', 'contents'),
          State('upload-data', 'filename'),
          State('upload-data', 'last_modified'),
          prevent_initial_call=True
          )
def get_model_prediction(n_clicks, image, filename, date):
    if n_clicks > 0:
        if image is not None:
            # Access the first uploaded file's content
            #image_content = image[0].split(",")[-1]  # Extract base64 content from data URL
            # Get the prediction
            prediction = predict_image(image)
            # Find the index of the max probability in the prediction array
            argmax_prediction = np.argmax(prediction)
            # Map the argmax prediction to class label using the class mapping dictionary
            predicted_class = class_mapping.get(argmax_prediction, 'Unknown')
            # Create an image element with processed image
            #processed_img_data = f'data:image/png;base64,{image_content}'
            # Return the predicted class label and the processed image
            #return html.Div(f'Predicted Class: {predicted_class}'), processed_img_data      
            model_output = html.Div([
                html.Div(f'Predicted Class: {predicted_class}')
            ])
            return [model_output]
    else:
        raise PreventUpdate
