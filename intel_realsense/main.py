import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import pyrealsense2 as rs

from segmentation_helper import SegmentationHelper
from annotation_writer import AnnotationWriter
from utils import frame_to_bgr_image
from camera_interface import CameraInterface

class RGBDCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB-D Data Collector")
        self.root.focus_force()

        self.cam = CameraInterface()
        self.cam.setup_streams()
        self.seg = SegmentationHelper(min_depth=300, max_depth=1200)
        self.writer = AnnotationWriter()

        base_path = Path("dataset")
        self.img_dir = base_path / "images"
        self.label_dir = base_path / "labels"
        self.depth_dir = base_path / "depth"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.counter = len(list(self.img_dir.glob("*.jpg")))

        self.captured_rgb = None
        self.captured_depth = None
        self.captured_mask = None
        self.is_capturing = True

        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.TOP)
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(side=tk.BOTTOM, pady=5)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()
        self.class_var = tk.IntVar(value=0)
        tk.Label(self.btn_frame, text="Class:").grid(row=1, column=0)
        self.class_selector = tk.OptionMenu(self.btn_frame, self.class_var, 0, 1)
        self.class_selector.grid(row=1, column=1)
        tk.Label(self.btn_frame, text="(0: Copper, 1: Steel)").grid(row=1, column=2, columnspan=2)

        self.capture_btn = tk.Button(self.btn_frame, text="Capture (Enter)", command=self.capture_frame)
        self.capture_btn.grid(row=0, column=0, padx=5)
        self.save_btn = tk.Button(self.btn_frame, text="Save (S)", command=self.save_data, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=1, padx=5)
        self.retake_btn = tk.Button(self.btn_frame, text="Retake (R)", command=self.retake_frame, state=tk.DISABLED)
        self.retake_btn.grid(row=0, column=2, padx=5)
        self.quit_btn = tk.Button(self.btn_frame, text="Quit (Q)", command=self.quit_app)
        self.quit_btn.grid(row=0, column=3, padx=5)

        self.root.bind('<Return>', lambda e: self.capture_frame())
        self.root.bind('s', lambda e: self.save_data())
        self.root.bind('S', lambda e: self.save_data())
        self.root.bind('r', lambda e: self.retake_frame())
        self.root.bind('R', lambda e: self.retake_frame())
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('Q', lambda e: self.quit_app())

        self.update_video()

    def update_video(self):
        try:
            if self.is_capturing:
                color_frame, depth_frame = self.cam.get_frames()
                if color_frame is not None and depth_frame is not None:
                    rgb = frame_to_bgr_image(color_frame)
                    preview = cv2.resize(rgb, (960, 540))
                    img = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"[ERROR] update_video failed: {e}")

        self.root.after(30, self.update_video)

    def capture_frame(self):
        if not self.is_capturing:
            return

        color_frame, depth_frame = self.cam.get_frames()
        if color_frame is None or depth_frame is None:
            print("[ERROR] No frame available to capture")
            return

        rgb = frame_to_bgr_image(color_frame)
        depth = np.asanyarray(depth_frame.get_data())

        mask = self.seg.segment(depth)

        self.captured_rgb = rgb
        self.captured_depth = depth
        self.captured_mask = mask

        mask_viz = (mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(mask_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_bgr, [largest], -1, (0, 255, 0), 2)

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        display_width, display_height = 320, 240
        rgb_resized = cv2.resize(rgb, (display_width, display_height))
        mask_resized = cv2.resize(mask_bgr, (display_width, display_height))
        depth_resized = cv2.resize(depth_colored, (display_width, display_height))

        combined = np.hstack((rgb_resized, mask_resized, depth_resized))
        cv2.putText(combined, "[Enter] Capture | [S] Save | [R] Retake | [Q] Quit", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.is_capturing = False
        self.capture_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)
        self.retake_btn.config(state=tk.NORMAL)
        print(f"[INFO] Frame captured for class: {self.class_var.get()}")

    def save_data(self):
        if self.captured_rgb is None or self.captured_mask is None:
            print("[WARNING] No frame to save.")
            return

        img_name = f"img{self.counter:04d}"
        cv2.imwrite(str(self.img_dir / f"{img_name}.jpg"), self.captured_rgb)
        np.save(str(self.depth_dir / f"{img_name}.npy"), self.captured_depth)
        selected_class = self.class_var.get()
        self.writer.write(
            str(self.label_dir / f"{img_name}.txt"),
            self.captured_mask,
            self.captured_rgb.shape[:2],
            label_class=selected_class
        )

        print(f"[SAVED] {img_name}")
        self.counter += 1
        self.reset_capture_state()

    def retake_frame(self):
        print("[RETAKE] Retaking frame.")
        self.reset_capture_state()

    def reset_capture_state(self):
        self.captured_rgb = None
        self.captured_depth = None
        self.captured_mask = None
        self.is_capturing = True
        self.capture_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.retake_btn.config(state=tk.DISABLED)

    def quit_app(self):
        print("[INFO] Quitting application.")
        self.cam.stop()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RGBDCollectorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")