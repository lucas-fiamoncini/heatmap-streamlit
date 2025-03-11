import streamlit as st
import cv2
import numpy as np
import imageio.v3 as iio
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# Helper function to resize image for display
def resize_image(image, max_width=800):
    """
    Resize the image to fit within a maximum width while maintaining aspect ratio.
    """
    height, width = image.shape[:2]
    if width > max_width:
        scale_factor = max_width / width
        new_height = int(height * scale_factor)
        new_width = max_width
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, scale_factor
    return image, 1.0

# Helper function to initialize session state
def initialize_session_state():
    if "images" not in st.session_state:
        st.session_state.images = []  # Stores uploaded images
        st.session_state.heatmaps = []  # Stores heatmaps for each image
        st.session_state.current_img_idx = 0  # Tracks the current image index
        st.session_state.brush_radius = 50  # Brush size
        st.session_state.brush_intensity = 0.5  # Brush intensity
        st.session_state.heatmap_histories = []  # Stores undo history for each heatmap
        st.session_state.scale_factors = []  # Stores scale factors for resized images

# Helper function to process uploaded images
def process_uploaded_files(uploaded_files):
    st.session_state.images = []
    st.session_state.heatmaps = []
    st.session_state.heatmap_histories = []
    st.session_state.scale_factors = []
    for file in uploaded_files:
        # Read image using imageio
        img = iio.imread(file)
        # Convert to 3-channel RGB if grayscale
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.session_state.images.append(img)
        # Initialize heatmap and history
        heatmap = np.zeros(img.shape[:2], dtype=np.float32)
        st.session_state.heatmaps.append(heatmap)
        st.session_state.heatmap_histories.append([heatmap.copy()])
        # Calculate scale factor for resizing
        resized_img, scale_factor = resize_image(img)
        st.session_state.scale_factors.append(scale_factor)

# Helper function to display the current image with heatmap overlay
def display_image_with_heatmap(image, heatmap, scale_factor):
    if heatmap.max() > 0:
        heatmap_normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    else:
        blended = image
    # Resize the blended image for display
    resized_blended, _ = resize_image(blended)
    return resized_blended

# Helper function to handle brush strokes
def apply_brush_stroke(heatmap, x, y, radius, intensity, scale_factor):
    # Scale coordinates back to original image size
    orig_x = int(x / scale_factor)
    orig_y = int(y / scale_factor)
    orig_radius = int(radius / scale_factor)
    cv2.circle(heatmap, (orig_x, orig_y), orig_radius, intensity, -1)

# Main Streamlit app
def main():
    st.title("Interactive Heatmap Painter 🔥")
    st.markdown("Upload images, draw heatmaps, and download the results!")

    # Initialize session state
    initialize_session_state()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload images (TIFF, PNG, JPG)",
        type=["tif", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    # Process uploaded files
    if uploaded_files and not st.session_state.images:
        process_uploaded_files(uploaded_files)

    # Display controls if images are uploaded
    if st.session_state.images:
        current_img_idx = st.session_state.current_img_idx
        current_image = st.session_state.images[current_img_idx]
        current_heatmap = st.session_state.heatmaps[current_img_idx]
        current_history = st.session_state.heatmap_histories[current_img_idx]
        scale_factor = st.session_state.scale_factors[current_img_idx]

        # Sidebar controls
        st.sidebar.header("Brush Settings")
        st.session_state.brush_radius = st.sidebar.slider(
            "Brush Size", 1, 100, st.session_state.brush_radius
        )
        st.session_state.brush_intensity = st.sidebar.select_slider(
            "Intensity", options=[0.5, 1.0], value=st.session_state.brush_intensity
        )

        # Display current image with heatmap overlay
        blended_image = display_image_with_heatmap(current_image, current_heatmap, scale_factor)
        st.image(blended_image, caption="Image with Heatmap Overlay", use_column_width=True)

        # Brush stroke input
        st.write("Click and drag to draw on the image:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",
            stroke_width=st.session_state.brush_radius,
            stroke_color="#FFFFFF",
            background_image=Image.fromarray(blended_image),
            height=blended_image.shape[0],
            width=blended_image.shape[1],
            drawing_mode="freedraw",
            key=f"canvas_{current_img_idx}",
        )

        # Apply brush strokes to heatmap
        if canvas_result.image_data is not None:
            stroke_mask = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
            ys, xs = np.where(stroke_mask > 0)
            for x, y in zip(xs, ys):
                apply_brush_stroke(
                    current_heatmap,
                    x,
                    y,
                    st.session_state.brush_radius,
                    st.session_state.brush_intensity,
                    scale_factor,
                )
            current_history.append(current_heatmap.copy())

        # Undo and Blur buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Undo") and len(current_history) > 1:
                current_history.pop()
                st.session_state.heatmaps[current_img_idx] = current_history[-1].copy()
                st.experimental_rerun()
        with col2:
            if st.button("Apply Gaussian Blur"):
                blurred_heatmap = cv2.GaussianBlur(current_heatmap, (51, 51), 25)
                current_history.append(blurred_heatmap.copy())
                st.session_state.heatmaps[current_img_idx] = blurred_heatmap
                st.experimental_rerun()

        # Image navigation
        st.divider()
        cols = st.columns([1, 2, 1])
        with cols[0]:
            if st.button("← Previous") and current_img_idx > 0:
                st.session_state.current_img_idx -= 1
                st.experimental_rerun()
        with cols[1]:
            st.markdown(f"**Image {current_img_idx + 1} of {len(st.session_state.images)}**")
        with cols[2]:
            if st.button("Next →") and current_img_idx < len(st.session_state.images) - 1:
                st.session_state.current_img_idx += 1
                st.experimental_rerun()

        # Download heatmap
        if current_heatmap.max() > 0:
            heatmap_normalized = (current_heatmap / current_heatmap.max() * 255).astype(np.uint8)
            heatmap_image = Image.fromarray(heatmap_normalized)
            buf = BytesIO()
            heatmap_image.save(buf, format="PNG")
            st.download_button(
                "💾 Download Heatmap",
                data=buf.getvalue(),
                file_name=f"heatmap_{current_img_idx}.png",
                mime="image/png",
            )
        else:
            st.warning("Draw on the image to create a heatmap.")

# Run the app
if __name__ == "__main__":
    main()