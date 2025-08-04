import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
import io

# ---------------- Noise Functions ----------------

def add_salt(img, salt):
    noisy_img = img.copy()
    salt_val = int(salt * img.size)
    for _ in range(salt_val):
        i = np.random.randint(0, img.shape[0])
        j = np.random.randint(0, img.shape[1])
        if len(img.shape) == 3 and img.shape[2] == 3:
            noisy_img[i, j] = [255, 255, 255]
        else:
            noisy_img[i, j] = 255
    return noisy_img

def add_pepper(img, pepper):
    noisy_img = img.copy()
    pepper_val = int(pepper * img.size)
    for _ in range(pepper_val):
        i = np.random.randint(0, img.shape[0])
        j = np.random.randint(0, img.shape[1])
        if len(img.shape) == 3 and img.shape[2] == 3:
            noisy_img[i, j] = [0, 0, 0]
        else:
            noisy_img[i, j] = 0
    return noisy_img

def add_gaussian_noise(img, mean=0, stddev=30):
    noise = np.random.normal(mean, stddev, img.shape)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# ---------------- Denoising ----------------

def remove_gaussian_noise(img):
    return cv.GaussianBlur(img, (3, 3), 0)

def remove_salt_pepper_noise(img):
    return cv.medianBlur(img, 3)

# ---------------- Filters ----------------

def apply_mean_filter(img, kernel_size):
    return cv.blur(img, (kernel_size, kernel_size))

def apply_gaussian_filter(img, kernel_size):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_median_filter(img, kernel_size):
    return cv.medianBlur(img, kernel_size)

# ---------------- Thresholding ----------------

def apply_threshold(img, threshold, max_value, method):
    if method == 'binary':
        _, result = cv.threshold(img, threshold, max_value, cv.THRESH_BINARY)
    elif method == 'trunc':
        _, result = cv.threshold(img, threshold, max_value, cv.THRESH_TRUNC)
    elif method == 'tozero':
        _, result = cv.threshold(img, threshold, max_value, cv.THRESH_TOZERO)
    elif method == 'binary_inv':
        _, result = cv.threshold(img, threshold, max_value, cv.THRESH_BINARY_INV)
    elif method == 'tozero_inv':
        _, result = cv.threshold(img, threshold, max_value, cv.THRESH_TOZERO_INV)
    else:
        raise ValueError("Invalid method.")
    return result

# ---------------- Morphology ----------------

def apply_morphology(img, operation, kernel_shape="rect"):
    if kernel_shape == "rect":
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    if operation == "Erosion":
        return cv.erode(img, kernel, iterations=1)
    elif operation == "Dilation":
        return cv.dilate(img, kernel, iterations=1)
    elif operation == "Opening":
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif operation == "Closing":
        return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    else:
        return img

# ---------------- Utility Functions ----------------

def ensure_rgb(img):
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    elif len(img.shape) == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img

def convert_to_gray(img):
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return img
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# ---------------- New Features ----------------

def convert_color_space(img, conversion):
    conversions = {
        "RGB to BGR": cv.COLOR_RGB2BGR,
        "BGR to RGB": cv.COLOR_BGR2RGB,
        "RGB to Grayscale": cv.COLOR_RGB2GRAY,
        "BGR to Grayscale": cv.COLOR_BGR2GRAY,
        "RGB to HSV": cv.COLOR_RGB2HSV,
        "HSV to RGB": cv.COLOR_HSV2RGB,
        "RGB to LAB": cv.COLOR_RGB2LAB,
        "LAB to RGB": cv.COLOR_LAB2RGB
    }
    return cv.cvtColor(img, conversions[conversion])

def resize_image(img, width, height):
    return cv.resize(img, (int(width), int(height)), interpolation=cv.INTER_AREA)

def add_text_simple(img, text, x, y, font_size=1):
    font = cv.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 2
    return cv.putText(img.copy(), text, (x, y), font, font_size, color, thickness, cv.LINE_AA)

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="Image Processing Tool", layout="centered")
st.title("Image Processing Tool")

if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'current_img' not in st.session_state:
    st.session_state.current_img = None
if 'is_grayscale' not in st.session_state:
    st.session_state.is_grayscale = False

if st.session_state.step == 'upload':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.current_img = np.array(image)
        st.session_state.is_grayscale = False
        st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Next"):
        st.session_state.step = 'process'
        st.rerun()

elif st.session_state.step == 'process':
    img = st.session_state.current_img

    display_img = img
    if st.session_state.is_grayscale and len(img.shape) == 2:
        display_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    st.image(display_img, caption="Current Image", use_container_width=True)

    option = st.selectbox("Choose Operation", [
        "Add Noise",
        "Remove Noise",
        "Apply Filter",
        "Segmentation",
        "Morphological Operation",
        "Change Color",
        "Resize Image",
        "Add Text"
    ])

    apply_button = st.button("Apply Operation")

    if option == "Add Noise":
        noise_type = st.selectbox("Noise Type", ["Salt", "Pepper", "Gaussian"])
        if noise_type == "Salt":
            amount = st.slider("Salt Amount", 0.0, 0.05, 0.01)
            if apply_button:
                st.session_state.current_img = add_salt(img, amount)
        elif noise_type == "Pepper":
            amount = st.slider("Pepper Amount", 0.0, 0.05, 0.01)
            if apply_button:
                st.session_state.current_img = add_pepper(img, amount)
        else:
            stddev = st.slider("Gaussian StdDev", 1, 100, 30)
            if apply_button:
                st.session_state.current_img = add_gaussian_noise(img, stddev=stddev)

    elif option == "Remove Noise":
        method = st.selectbox("Method", ["Gaussian", "Salt & Pepper"])
        if apply_button:
            if method == "Gaussian":
                st.session_state.current_img = remove_gaussian_noise(img)
            else:
                st.session_state.current_img = remove_salt_pepper_noise(img)

    elif option == "Apply Filter":
        filter_type = st.selectbox("Filter Type", ["Mean", "Gaussian", "Median"])
        kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2)
        if apply_button:
            if filter_type == "Mean":
                st.session_state.current_img = apply_mean_filter(img, kernel_size)
            elif filter_type == "Gaussian":
                st.session_state.current_img = apply_gaussian_filter(img, kernel_size)
            else:
                st.session_state.current_img = apply_median_filter(img, kernel_size)

    elif option == "Segmentation":
        method = st.selectbox("Method", ["Global", "Adaptive", "Otsu"])
        gray_img = convert_to_gray(img)

        if method == "Global":
            threshold = st.slider("Threshold", 0, 255, 127)
            max_val = st.slider("Max Value", 1, 255, 255)
            mode = st.selectbox("Type", ["binary", "trunc", "tozero", "binary_inv", "tozero_inv"])
            if apply_button:
                result = apply_threshold(gray_img, threshold, max_val, mode)
                st.session_state.current_img = result
                st.session_state.is_grayscale = True

        elif method == "Adaptive":
            block_size = st.slider("Block Size", 3, 255, 11, step=2)
            C = st.slider("C", 0, 20, 2)
            if apply_button:
                result = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C)
                st.session_state.current_img = result
                st.session_state.is_grayscale = True

        else:
            if apply_button:
                _, result = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                st.session_state.current_img = result
                st.session_state.is_grayscale = True

    elif option == "Morphological Operation":
        if not st.session_state.is_grayscale:
            st.error("You must apply thresholding before morphological operations.")
        else:
            morph_op = st.selectbox("Morph Operation", ["Erosion", "Dilation", "Opening", "Closing"])
            kernel_shape = st.radio("Kernel Shape", ["rect", "cross"])
            if apply_button:
                st.session_state.current_img = apply_morphology(img, morph_op, kernel_shape)

    elif option == "Change Color":
        color_conversion = st.selectbox("Select Conversion", [
            "RGB to BGR", "BGR to RGB", "RGB to Grayscale",
            "BGR to Grayscale", "RGB to HSV", "HSV to RGB",
            "RGB to LAB", "LAB to RGB"
        ])
        if apply_button:
            st.session_state.current_img = convert_color_space(img, color_conversion)
            if "Grayscale" in color_conversion:
                st.session_state.is_grayscale = True
            else:
                st.session_state.is_grayscale = False

    elif option == "Resize Image":
        new_width = st.number_input("New Width", min_value=10, value=img.shape[1])
        new_height = st.number_input("New Height", min_value=10, value=img.shape[0])
        if apply_button:
            st.session_state.current_img = resize_image(img, new_width, new_height)

    elif option == "Add Text":
        text = st.text_input("Text to Add", "Hello, OpenCV!")
        x = st.slider("X Position", 0, img.shape[1], 50)
        y = st.slider("Y Position", 0, img.shape[0], 50)
        font_size = st.slider("Font Size", 0.5, 3.0, 1.0)
        if apply_button:
            st.session_state.current_img = add_text_simple(img, text, x, y, font_size)

    if st.button("Reset Image"):
        st.session_state.step = 'upload'
        st.session_state.current_img = None
        st.session_state.is_grayscale = False
        st.rerun()

    updated_img = st.session_state.current_img
    if st.session_state.is_grayscale and len(updated_img.shape) == 2:
        display_updated = cv.cvtColor(updated_img, cv.COLOR_GRAY2RGB)
    else:
        display_updated = updated_img
    st.image(display_updated, caption="Updated Image", use_container_width=True)

    if st.session_state.is_grayscale and len(updated_img.shape) == 2:
        img_pil = Image.fromarray(updated_img).convert('L')
    else:
        if len(updated_img.shape) == 3 and updated_img.shape[2] == 3:
            img_pil = Image.fromarray(updated_img)
        else:
            img_pil = Image.fromarray(updated_img).convert('RGB')

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Image", data=byte_im, file_name="processed_image.png", mime="image/png")