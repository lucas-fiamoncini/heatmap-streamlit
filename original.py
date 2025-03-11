import cv2
import numpy as np
import imageio.v3 as iio
import os


class HeatmapPainter:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.current_img_idx = 0
        self.brush_radius = 50  # Initial brush size
        self.brush_intensity = 0.5  # Initial intensity (0.5 or 1)
        self.heatmap_history = []  # Store heatmap states for undo
        self.setup_image()

    def setup_image(self):
        # Read image with imageio (supports TIFF)
        self.current_image_path = self.image_paths[self.current_img_idx]
        self.original_image = iio.imread(self.current_image_path)

        # Convert to BGR format for OpenCV
        if self.original_image.ndim == 2:  # Grayscale image
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        else:  # Color image (assumed RGB)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)

        # Initialize heatmap
        self.heatmap = np.zeros_like(self.original_image[..., 0], dtype=np.float32)
        self.display_image = self.resize_image_for_display(self.original_image)
        self.display_heatmap = np.zeros_like(self.display_image[..., 0], dtype=np.float32)
        self.heatmap_history = [self.heatmap.copy()]  # Initialize history

    def resize_image_for_display(self, image):
        # Resize the image to fit the notebook screen
        height, width = image.shape[:2]
        max_height = 600  # Adjust this value based on your screen size
        if height > max_height:
            scale_factor = max_height / height
            new_width = int(width * scale_factor)
            new_height = max_height
            return cv2.resize(image, (new_width, new_height))
        return image

    def update_display(self):
        # Resize the heatmap to match the display image size
        self.display_heatmap = cv2.resize(self.heatmap, (self.display_image.shape[1], self.display_image.shape[0]))

        # Overlay heatmap on the image with transparency
        heatmap_colored = cv2.applyColorMap(
            (self.display_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        alpha = 0.5  # Heatmap transparency
        display = cv2.addWeighted(self.display_image, 1 - alpha, heatmap_colored, alpha, 0)
        cv2.imshow("Paint Heatmap", display)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # Save current heatmap state to history
            self.heatmap_history.append(self.heatmap.copy())

            # Calculate the coordinates in the original image
            scale_x = self.original_image.shape[1] / self.display_image.shape[1]
            scale_y = self.original_image.shape[0] / self.display_image.shape[0]
            original_x = int(x * scale_x)
            original_y = int(y * scale_y)

            # Add intensity to the heatmap where the mouse is dragged
            cv2.circle(
                self.heatmap, (original_x, original_y), self.brush_radius, self.brush_intensity, -1
            )
            self.update_display()

    def undo_last_change(self):
        if len(self.heatmap_history) > 1:  # Ensure there's a previous state
            self.heatmap_history.pop()  # Remove current state
            self.heatmap = self.heatmap_history[-1].copy()  # Revert to previous state
            self.update_display()

    def save_heatmap(self):
        # Extract the base name of the image file (without extension)
        image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        heatmap_filename = f"heatmap_{image_name}.png"

        # Normalize and save the heatmap
        heatmap_normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        cv2.imwrite(heatmap_filename, heatmap_normalized)
        print(f"Heatmap saved as {heatmap_filename}")
        self.show_warning(f"Heatmap saved as {heatmap_filename}")

    def show_warning(self, message):
        # Display a warning message on the image
        warning_image = self.display_image.copy()
        cv2.putText(
            warning_image,
            message,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Paint Heatmap", warning_image)
        cv2.waitKey(1000)  # Show the warning for 1 second
        self.update_display()

    def run(self):
        cv2.namedWindow("Paint Heatmap")
        cv2.setMouseCallback("Paint Heatmap", self.mouse_callback)

        # Create trackbars for brush settings
        cv2.createTrackbar(
            "Brush Size", "Paint Heatmap", self.brush_radius, 100, lambda x: None
        )
        cv2.createTrackbar(
            "Intensity",
            "Paint Heatmap",
            0,  # Start at position 0 (0.5 intensity)
            1,  # Only two positions (0 and 1)
            lambda x: None,
        )

        while True:
            self.brush_radius = cv2.getTrackbarPos("Brush Size", "Paint Heatmap")
            intensity_pos = cv2.getTrackbarPos("Intensity", "Paint Heatmap")
            self.brush_intensity = 0.5 if intensity_pos == 0 else 1.0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):  # Next image
                # Save current heatmap before moving to the next image
                self.save_heatmap()

                # Move to the next image
                self.current_img_idx += 1
                if self.current_img_idx >= len(self.image_paths):  # End of list
                    print("All images processed. Closing windows.")
                    cv2.destroyAllWindows()
                    break
                self.setup_image()
                self.update_display()
            elif key == ord("p"):  # Previous image
                # Move to the previous image
                self.current_img_idx -= 1
                if self.current_img_idx < 0:  # Start of list
                    self.current_img_idx = 0
                    print("Already at the first image.")
                self.setup_image()
                self.update_display()
            elif key == ord("s"):  # Save heatmap
                self.save_heatmap()
            elif key == ord("u"):  # Undo last change
                self.undo_last_change()
            elif key == ord("x"):  # Apply Gaussian blur
                self.heatmap_history.append(
                    self.heatmap.copy()
                )  # Save state before blur
                self.heatmap = cv2.GaussianBlur(self.heatmap, (51, 51), 25)
                self.update_display()
            elif key == 27:  # Exit on ESC
                break

        cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    # image_paths = ["image003-2-roi1.tif", "image003-2-roi2.tif"]  # Replace with your image paths
    image_paths = [
        "C:\\Users\\lucas\\LUCAS\\Datasets\\OED-Mestrado\\Imagens\\032\\TC_28.tif",
        "C:\\Users\\lucas\\LUCAS\\Datasets\\OED-Mestrado\\Imagens\\032\\TC_30.tif",
        "C:\\Users\\lucas\\LUCAS\\Datasets\\OED-Mestrado\\Imagens\\032\\TC_31.tif",
        "C:\\Users\\lucas\\LUCAS\\Datasets\\OED-Mestrado\\Imagens\\097\\TC_28.tif",
        "C:\\Users\\lucas\\LUCAS\\Datasets\\OED-Mestrado\\Imagens\\097\\TC_29.tif",
    ]
    painter = HeatmapPainter(image_paths)
    painter.run()