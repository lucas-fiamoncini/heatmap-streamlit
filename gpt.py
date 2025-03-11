import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io

# --- Helper Functions ---

def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV image (BGR)."""
    open_cv_image = np.array(image.convert("RGB"))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV image (BGR) to a PIL image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def apply_heatmap_overlay(original_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Resize the heatmap to the original image size and overlay it with transparency.
    The heatmap is assumed to have values in the range [0,1].
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = 0.5  # Adjust the transparency here
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

# --- Main Application Function ---

def main():
    # Initialize Streamlit session state for heatmap and history if not already defined
    if "heatmap" not in st.session_state:
        st.session_state.heatmap = None
    if "heatmap_history" not in st.session_state:
        st.session_state.heatmap_history = []
    
    st.title("Heatmap Painter App")
    
    # --- Image Upload ---
    uploaded_files = st.file_uploader(
        "Upload one or more images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True
    )
    
    if uploaded_files:
        # For demonstration, use the first uploaded image
        file = uploaded_files[0]
        image = Image.open(file)
        original_img = pil_to_cv(image)
        st.session_state.original_img = original_img.copy()
    
        # Initialize heatmap if not already set
        if st.session_state.heatmap is None:
            st.session_state.heatmap = np.zeros(original_img.shape[:2], dtype=np.float32)
            st.session_state.heatmap_history = [st.session_state.heatmap.copy()]
    
        # --- Sidebar Controls ---
        st.sidebar.header("Brush Settings")
        brush_radius = st.sidebar.slider("Brush Size", min_value=1, max_value=100, value=50)
        intensity_option = st.sidebar.radio("Intensity", options=["0.5", "1.0"])
        intensity = 0.5 if intensity_option == "0.5" else 1.0
    
        # --- Canvas for Drawing ---
        st.write("Draw on the image to add heat (use free-draw mode)")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Semi-transparent red for drawing
            stroke_width=brush_radius,
            stroke_color="#ff0000",
            background_image=Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=600,
            width=int(original_img.shape[1] * 600 / original_img.shape[0]),
            drawing_mode="freedraw",
            key="canvas",
        )
    
        # --- Process Canvas Drawing ---
        if canvas_result.image_data is not None:
            # Convert the canvas image (RGBA) to grayscale to create a mask
            drawing = canvas_result.image_data.astype(np.uint8)
            drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_RGBA2GRAY)
            _, mask = cv2.threshold(drawing_gray, 10, 255, cv2.THRESH_BINARY)
            # Resize the mask to match the original image dimensions
            mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
            # Normalize mask to range [0,1] and scale by the selected intensity
            mask_normalized = (mask_resized.astype(np.float32) / 255.0) * intensity
            # Save the current heatmap state for undo functionality
            st.session_state.heatmap_history.append(st.session_state.heatmap.copy())
            # Add the drawn mask to the existing heatmap and clip values to [0,1]
            st.session_state.heatmap = np.clip(st.session_state.heatmap + mask_normalized, 0, 1)
    
        # --- Display Heatmap Overlay ---
        overlay = apply_heatmap_overlay(original_img, st.session_state.heatmap)
        st.image(cv_to_pil(overlay), caption="Heatmap Overlay", use_column_width=True)
    
        # --- Action Buttons ---
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Undo"):
            if len(st.session_state.heatmap_history) > 1:
                st.session_state.heatmap_history.pop()  # Remove current state
                st.session_state.heatmap = st.session_state.heatmap_history[-1].copy()
        if col2.button("Apply Gaussian Blur"):
            st.session_state.heatmap_history.append(st.session_state.heatmap.copy())
            st.session_state.heatmap = cv2.GaussianBlur(st.session_state.heatmap, (51, 51), 25)
        if col3.button("Reset Heatmap"):
            st.session_state.heatmap_history.append(st.session_state.heatmap.copy())
            st.session_state.heatmap = np.zeros(original_img.shape[:2], dtype=np.float32)
        if col4.button("Save Heatmap"):
            # Normalize the heatmap and save as a PNG file
            heatmap_norm = np.uint8((st.session_state.heatmap / (st.session_state.heatmap.max() + 1e-6)) * 255)
            heatmap_filename = f"heatmap_{file.name.split('.')[0]}.png"
            cv2.imwrite(heatmap_filename, heatmap_norm)
            st.success(f"Heatmap saved as {heatmap_filename}")
            # Prepare a download button for the saved heatmap
            _, im_buf_arr = cv2.imencode('.png', heatmap_norm)
            st.download_button(
                "Download Heatmap",
                data=im_buf_arr.tobytes(),
                file_name=heatmap_filename,
                mime="image/png",
            )

if __name__ == "__main__":
    main()
