import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('cat-breeds-model-augment.hdf5')
  return model


with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Cat Breed Classification
         """
         )

file = st.file_uploader("Please upload an image", type=["jpg", "png"])

from PIL import Image, ImageOps
import numpy as np

st.sidebar.header('About')
logo_url = "https://streamlit.io/images/brand/streamlit-mark-color.svg"
st.sidebar.image(logo_url, width=100)

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        #size = (180,180)    
        #image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        #image = np.asarray(image)
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #img_reshape = img[np.newaxis,...]

        img_array = tf.keras.utils.img_to_array(image_data)
        img_array = tf.expand_dims(img_array, 0)
    
        prediction = model.predict(img_array)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image = image.resize((180,180))
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    #st.write(predictions)
    #st.write("SCORE:")
    arr = np.argsort(score)[-3:][::-1]
    #st.write(arr)
    
    class_names=['bengal', 'british', 'maine_coon', 'ragdoll', 'scottish_fold']
    st.header("Top 3 results:")
    st.subheader("1. " + class_names[arr[0]] + " cat")
    st.subheader("2. " + class_names[arr[1]] + " cat")
    st.subheader("3. " + class_names[arr[2]] + " cat")
    print(
    "This image most likely belongs to **{}** with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    st.write("""This image most likely belongs to **{}** with a {:.2f} percent confidence.
    """
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
