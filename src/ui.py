import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import yaml
from collections import deque


class RealtimeHeadChart(ttk.Frame):
    """Biểu đồ realtime hiển thị Roll, Pitch, Yaw với thiết kế đẹp hơn."""
    def __init__(self, parent, max_points=50):
        super().__init__(parent)
        self.max_points = max_points
        self.roll_data = deque(maxlen=max_points)
        self.pitch_data = deque(maxlen=max_points)
        self.yaw_data = deque(maxlen=max_points)
        self.x_data = deque(maxlen=max_points)
        self.counter = 0

        # Tạo figure với style đẹp hơn
        self.fig = Figure(figsize=(5, 3), dpi=100, facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111, facecolor='#ffffff')
        self.ax.set_title("Head Orientation (°)", fontsize=11, fontweight='bold', pad=10)
        self.ax.set_xlabel("Frames", fontsize=9)
        self.ax.set_ylabel("Degrees", fontsize=9)
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Đường với màu sắc đẹp hơn
        self.line_roll, = self.ax.plot([], [], "#F75252", linewidth=2, label='Roll', alpha=0.9)
        self.line_pitch, = self.ax.plot([], [], '#4CAF50', linewidth=2, label='Pitch', alpha=0.9)
        self.line_yaw, = self.ax.plot([], [], '#2196F3', linewidth=2, label='Yaw', alpha=0.9)
        
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_data(self, head_orientation):
        """Cập nhật dữ liệu mới vào biểu đồ"""
        if not head_orientation:
            return
        self.counter += 1
        self.x_data.append(self.counter)
        self.roll_data.append(head_orientation.get("roll", 0))
        self.pitch_data.append(head_orientation.get("pitch", 0))
        self.yaw_data.append(head_orientation.get("yaw", 0))

        self.line_roll.set_data(self.x_data, self.roll_data)
        self.line_pitch.set_data(self.x_data, self.pitch_data)
        self.line_yaw.set_data(self.x_data, self.yaw_data)

        self.ax.set_xlim(max(0, self.counter - self.max_points), self.counter)
        self.ax.set_ylim(-40, 40)

        self.canvas.draw()
        self.canvas.flush_events()


class AlertPanel(tk.Frame):
    """Panel hiển thị cảnh báo với hiệu ứng nhấp nháy"""
    def __init__(self, parent):
        super().__init__(parent, bg='#1a1a1a', height=80)
        self.pack_propagate(False)
        
        self.is_alert = False
        self.blink_state = False
        
        # Icon và text
        self.alert_label = tk.Label(
            self,
            text="⚠️ CẢNH BÁO BUỒN NGỦ ⚠️",
            font=('Arial', 18, 'bold'),
            fg='#FF5252',
            bg='#1a1a1a'
        )
        self.alert_label.pack(expand=True)
        
        self.status_label = tk.Label(
            self,
            text="● TRẠNG THÁI: Bình thường",
            font=('Arial', 11),
            fg='#4CAF50',
            bg='#1a1a1a'
        )
        self.status_label.pack(pady=(0, 5))
        
        self._start_blink_animation()
    
    def set_alert(self, is_alert):
        """Kích hoạt/tắt cảnh báo"""
        self.is_alert = is_alert
        if not is_alert:
            self.config(bg='#1a1a1a')
            self.alert_label.config(bg='#1a1a1a', fg='#4CAF50')
            self.status_label.config(bg='#1a1a1a', fg='#4CAF50', text="● TRẠNG THÁI: Bình thường")
    
    def _start_blink_animation(self):
        """Hiệu ứng nhấp nháy khi có cảnh báo"""
        if self.is_alert:
            self.blink_state = not self.blink_state
            if self.blink_state:
                self.config(bg='#FF5252')
                self.alert_label.config(bg='#FF5252', fg='#FFFFFF')
                self.status_label.config(bg='#FF5252', fg='#FFFFFF', text="⚠️ TRẠNG THÁI: NGUY HIỂM!")
            else:
                self.config(bg='#FF1744')
                self.alert_label.config(bg='#FF1744', fg='#FFEB3B')
                self.status_label.config(bg='#FF1744', fg='#FFEB3B', text="⚠️ TRẠNG THÁI: NGUY HIỂM!")
        
        self.after(300, self._start_blink_animation)


class ConfigManager:
    def __init__(self, yaml_path="configs/configs.yaml"):
        self.yaml_path = yaml_path
        self.config = {
            "led_pin": 17,
            "model_path": "face_landmarker.task",
            "blink_threshold_wo_pitch": 0.15,
            "blink_threshold_pitch": 0.25,
            "pitch_threshold_negative": -5.0,
            "pitch_threshold_positive": 5.0,
            "delay_drowsy_threshold": 3.0,
            "perclos_threshold": 0.8,
            "frame_height": 720,
            "frame_width": 1280,
            "roi_coords": None
        }
        try:
            self.load()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CONFIG] Error loading config: {e}")

    def load(self):
        with open(self.yaml_path, "r") as f:
            self.config.update(yaml.safe_load(f))

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def save(self):
        try:
            with open(self.yaml_path, "w") as f:
                yaml.safe_dump(self.config, f)
        except Exception as e:
            print(f"[CONFIG] WARNING: Could not save config: {e}")


