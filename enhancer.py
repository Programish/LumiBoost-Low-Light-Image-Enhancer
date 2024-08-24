# import streamlit as st
# from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the model
# model = keras.models.load_model('enhance.tf')
# print(type(model))
# # Function to enhance the image
# def infer(original_image):
#     image = keras.preprocessing.image.img_to_array(original_image)
#     image = image[:, :, :3] if image.shape[-1] > 3 else image
#     image = image.astype("float32") / 255.0
#     image = np.expand_dims(image, axis=0)
#     output_image = model(image)
#     output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
#     output_image = Image.fromarray(output_image.numpy())
#     return output_image

# # Function to plot the results
# def plot_results(images, titles, figure_size=(12, 12)):
#     fig = plt.figure(figsize=figure_size)
#     for i in range(len(images)):
#         fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
#         _ = plt.imshow(images[i])
#         plt.axis("off")
#     plt.show()

# # Streamlit code to upload the file
# uploaded_file = st.file_uploader("Choose an image...", type="png")
# if st.button('Enhance Image'):
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Processing...")

#     # Generate enhanced light images using the reloaded model
#     enhanced_image = infer(image)

#     # Display the original and enhanced images
#     plot_results(
#         [image, enhanced_image],
#         ["Original_Image", "Enhanced_Image"],
#         (15, 7),
#     )
#     st.image(enhanced_image)

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load the model
model = keras.models.load_model('enhance.tf')
# model = tf.keras.models.load_model('enhance')
# model= keras.layers.TFSMLayer('enhance.tf', call_endpoint='serving_default')
# print(type(model))
# Function to enhance the image
def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image[:, :, :3] if image.shape[-1] > 3 else image
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = model(image)
    # print(output_image)
    # Load and preprocess the image
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    
    return output_image

# Function to plot the results
def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()

    
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                # render_dashboard(user)
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None
def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")
        
def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


# def render_dashboard(user_info, json_file_path="data.json"):
#     try:
#         st.title(f"Welcome to the Dashboard, {user_info['name']}!")
#         st.subheader("User Information:")
#         st.write(f"Name: {user_info['name']}")
#         st.write(f"Sex: {user_info['sex']}")
#         st.write(f"Age: {user_info['age']}")

#     except Exception as e:
#         st.error(f"Error rendering dashboard: {e}")
      
def main(json_file_path="data.json"):
    st.sidebar.title("Low Light")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "AboutUs" , 'Image Enhancer'),
        key="Image Enhancer",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)
            
    elif page =="AboutUs":
         st.title("About Us")
         st.markdown(
        """
        We are a team passionate about photography and technology. Our goal is to
        help users enhance their photos, especially those taken in low-light conditions,
        to make them more visually appealing and memorable.

        Our Low Light Image Enhancer app uses advanced image processing algorithms
        to improve brightness, reduce noise, and enhance overall image quality. We
        strive to provide an easy-to-use and effective tool for photographers and
        photo enthusiasts alike.

        Thank you for using our app. We hope you enjoy using it as much as we enjoyed
        creating it!
        """
    )

    elif page == "Image Enhancer":
        if session_state.get("logged_in"):
            st.title("Image Enhancer")
            # Streamlit code to upload the file
            uploaded_file = st.file_uploader("Choose an image...", type="png")
            if st.button('Enhance Image'):
                image = Image.open(uploaded_file)
                image = image.resize((600, 400))
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write("")
                st.write("Processing...")

                # Generate enhanced light images using the reloaded model
                enhanced_image = infer(image)

                # Display the original and enhanced images
                plot_results(
                    [image, enhanced_image],
                    ["Original_Image", "Enhanced_Image"],
                    (15, 7),
                )
                st.image(enhanced_image)
                
        else:
            st.warning("Please login/signup to use the app")
            
initialize_database()
main()