import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Disable GPU and use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

def main():
    st.title('Sc-islet functionality predictor')
    st.write('Upload an image of your sc-islets to predict functionality')

    file = st.file_uploader('Upload an image of sc-islets', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Resize image to 256x256 and normalize
        resized_image = image.resize((256, 256))
        img_array = np.array(resized_image) / 255.0  # Ensure correct normalization
        img_array = img_array.reshape((1, 256, 256, 3))

        # Debugging: Check the input image array
        st.write(f"Image shape: {img_array.shape}")
        st.write(f"Image min value: {np.min(img_array)}, max value: {np.max(img_array)}")

        # Load pre-trained model
        model = tf.keras.models.load_model('/home/el/Documents/workspace/Image_classifier/sc-islet_classifier/models/VIT_imageclassifier.h5')

        # Predict functionality
        prediction = model.predict(img_array)  # Get the prediction
        st.write(f"Model prediction raw output: {prediction}")  # Debugging: See the raw output
        prediction = prediction[0][0]  # Assuming single probability between 0 and 1

        non_functional_prob = 1 - prediction  # Probability of being "Non-functional"
        functional_prob = prediction  # Probability of being "Functional"
        
        # Class labels for reference
        classes = ['Non-functional', 'Functional']
        probabilities = [non_functional_prob, functional_prob]

        # Clear the current figure to avoid overlap
        plt.clf()

        # Create the plot for comparison
        fig, ax = plt.subplots()
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, probabilities, align='center', color=['orange', 'blue'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()  # Highest probability on top
        ax.set_xlabel('Probability')
        ax.set_title('Functionality prediction')

        # Display the plot
        st.pyplot(fig)

        # Output the predicted class and its probability
        predicted_class = classes[int(round(functional_prob))]
        st.write(f"Predicted class: **{predicted_class}** with probability: **{functional_prob:.2f}**")
    else:
        st.text('You have not uploaded an image yet.')

if __name__ == "__main__":
    main()

