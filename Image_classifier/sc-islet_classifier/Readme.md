# sc-islet Functionality Classifier

This project classifies iPSC-derived sc-islets into functional or non-functional categories based on brightfield images using a Transformer-based model.

## Overview

The project includes a Jupyter notebook for training the model and a Streamlit application for user interaction. Users can upload images of sc-islets, and the app will predict whether they are functional or non-functional.

## Setup and Installation

### Requirements

Make sure you have the following packages installed:

- Python 3.x
- TensorFlow
- Streamlit
- NumPy
- Matplotlib
- Pillow

You can install the required packages using `pip`:

```bash
pip install tensorflow streamlit numpy matplotlib pillow
```
### Streamlit app
```bash
Run the Streamlit application:
```
###Project Structure
- transformer_classifier.ipynb: The main Jupyter notebook containing the model training code for classifying sc-islets.
- streamlit_app.py: The Streamlit application for predicting sc-islet functionality based on uploaded images.

### Results

After running the Streamlit app, you'll see the predicted functionality of the sc-islets, along with the probability of the prediction. The result will be displayed as a bar chart showing the probabilities for "Functional" and "Non-functional" classes.
Contributing

### Contact

For any questions or suggestions, feel free to reach out:

Name: Elhadi Iich
Email: e.iich@hotmail.nl
