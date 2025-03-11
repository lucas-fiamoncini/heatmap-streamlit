import streamlit as st
import cv2
import numpy as np
import imageio.v3 as iio
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# Initialize session state
def init_state():
    if "initialized" not in st.session_state:
        st.session_state.update({
            "initialized": True,
            "images": [],
            "heatmaps": [],
            "current_idx": 0,
            "brush_size": 30,  # Initialize brush_size
            "brush_intensity": 1.0,  # Initialize brush_intensity
            "history": [],
            "scale_factors": [],
            "original_shapes": []
        })

# Process uploaded files
def process_files(uploaded_files):
    st.session_state.images = []
    st.session_state.heatmaps = []
    st.session_state.history = []
    st.session_state.scale_factors = []
    st.session_state.original_shapes = []
    
    for file in uploaded_files:
        # Read and convert image
        img = iio.imread(file)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        h, w = img.shape[:2]
        st.session_state.original_shapes.append((h, w))
        
        # Create resized display version
        display_img, scale = resize_image(img)
        st.session_state.images.append(display_img)
        st.session_state.scale_factors.append(scale)
        
        # Initialize heatmap and history
        heatmap = np.zeros((h, w), dtype=np.float32)
        st.session_state.heatmaps.append(heatmap)
        st.session_state.history.append([heatmap.copy()])

def resize_image(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (max_width, int(h * scale))), scale
    return img, 1.0

def apply_blur(heatmap):
    """Preserve intensity while smoothing edges"""
    current_max = heatmap.max()
    blurred = cv2.GaussianBlur(heatmap, (51, 51), 25)
    return blurred * (current_max / blurred.max()) if current_max > 0 else blurred

def main():
    st.title("Medical Image Heatmap Painter")
    init_state()  # Ensure session state is initialized
    
    # File upload
    uploaded_files = st.file_uploader("Upload images", 
                                    type=["tif", "png", "jpg", "jpeg"],
                                    accept_multiple_files=True)
    
    if uploaded_files and not st.session_state.images:
        process_files(uploaded_files)

    if st.session_state.images:
        current_idx = st.session_state.current_idx
        display_img = st.session_state.images[current_idx]
        heatmap = st.session_state.heatmaps[current_idx]
        scale = st.session_state.scale_factors[current_idx]
        original_shape = st.session_state.original_shapes[current_idx]

        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            st.session_state.brush_size = st.slider("Brush Size", 1, 100, st.session_state.brush_size)
            st.session_state.brush_intensity = st.select_slider("Intensity", 
                                                              options=[0.5, 1.0], 
                                                              value=st.session_state.brush_intensity)
            
            if st.button("Apply Gaussian Blur"):
                new_heatmap = apply_blur(heatmap)
                st.session_state.history[current_idx].append(heatmap.copy())
                st.session_state.heatmaps[current_idx] = new_heatmap
                st.rerun()

            if st.button("Undo") and len(st.session_state.history[current_idx]) > 1:
                st.session_state.history[current_idx].pop()
                st.session_state.heatmaps[current_idx] = st.session_state.history[current_idx][-1].copy()
                st.rerun()

        # Create overlay
        heatmap_display = cv2.resize(heatmap, 
                                    (display_img.shape[1], display_img.shape[0]))
        heatmap_colored = cv2.applyColorMap((heatmap_display * 255).astype(np.uint8), 
                                           cv2.COLORMAP_JET)
        blended = cv2.addWeighted(display_img, 0.5, heatmap_colored, 0.5, 0)

        # Drawing canvas
        canvas = st_canvas(
            stroke_width=st.session_state.brush_size,
            stroke_color="#e4eded",
            background_image=Image.fromarray(blended),
            height=blended.shape[0],
            width=blended.shape[1],
            drawing_mode="freedraw",
            key=f"canvas_{current_idx}",
        )

        # Process drawing
        if canvas.image_data is not None:
            mask = cv2.cvtColor(np.array(canvas.image_data), cv2.COLOR_RGBA2GRAY)
            ys, xs = np.where(mask > 10)
            
            if len(xs) > 0:
                # Save current state
                st.session_state.history[current_idx].append(heatmap.copy())
                
                # Convert coordinates to original scale
                scale_x = original_shape[1] / blended.shape[1]
                scale_y = original_shape[0] / blended.shape[0]
                
                for x, y in zip(xs, ys):
                    orig_x = int(x * scale_x)
                    orig_y = int(y * scale_y)
                    # radius = int(st.session_state.brush_size * scale_x)
                    radius = st.session_state.brush_size
                    
                    cv2.circle(heatmap, 
                             (orig_x, orig_y),
                             radius,
                             st.session_state.brush_intensity,
                             -1)
                
                st.rerun()

        # Navigation
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("← Previous") and current_idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()
        with col2:
            if st.button("Next →") and current_idx < len(st.session_state.images)-1:
                st.session_state.current_idx += 1
                st.rerun()

        # Download
        if heatmap.max() > 0:
            heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
            img = Image.fromarray(heatmap_norm)
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.download_button("Download Heatmap", 
                             data=buf.getvalue(),
                             file_name="heatmap.png",
                             mime="image/png")

if __name__ == "__main__":
    main()