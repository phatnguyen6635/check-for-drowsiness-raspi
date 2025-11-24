import os
import socket
import requests
from datetime import datetime
from collections import defaultdict


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"

class ClientUploader:
    def __init__(self, folder: str, server_url: str):
        self.image_folder = folder + '/log_frame'
        self.log_folder = folder + '/log_app'
        self.upload_endpoint = f"{server_url.rstrip('/')}/upload"
        self.log_endpoint = f"{server_url.rstrip('/')}/log"

        self.hostname = socket.gethostname()
        self.ip = get_ip()
        self.host_ip = f"{self.hostname}@{self.ip}"

        os.makedirs(folder, exist_ok=True)

    def extract(self, filename: str):
        name, _ = os.path.splitext(filename)
        parts = name.split("_")
        if len(parts) < 4:
            return None, None
        file_type = parts[1]  
        timestamp = parts[2] + "_" + parts[3]
        return file_type, timestamp

    def group_files(self):
        groups = defaultdict(dict)
        for f in os.listdir(self.image_folder):
            if not f.lower().endswith(".jpg"):
                continue
            file_type, ts = self.extract(f)
            if not file_type or not ts:
                continue
            if file_type in ("origin", "annotated"):
                groups[ts][file_type] = f
        return groups

    def send_images(self):
        groups = self.group_files()

        if not groups:
            print("[SEND] No images found.")
            return

        for ts, files in groups.items():
            if "origin" not in files or "annotated" not in files:
                print(f"[SKIP] Missing pair for timestamp {ts}")
                continue  

            origin_file = files["origin"]
            annotated_file = files["annotated"]

            new_origin_name = f"{self.hostname}_{self.ip}_{ts}_origin.jpg"
            new_annotated_name = f"{self.hostname}_{self.ip}_{ts}_annotated.jpg"

            try:
                with open(os.path.join(self.image_folder, origin_file), "rb") as f:
                    r1 = requests.post(
                        self.upload_endpoint,
                        files={"file": (new_origin_name, f, "image/jpeg")},
                        timeout=10
                    )
                with open(os.path.join(self.image_folder, annotated_file), "rb") as f:
                    r2 = requests.post(
                        self.upload_endpoint,
                        files={"file": (new_annotated_name, f, "image/jpeg")},
                        timeout=10
                    )

                if r1.status_code == 200 and r2.status_code == 200:
                    print(f"[SEND] Uploaded pair {ts} OK → deleting local files")
                    os.remove(os.path.join(self.image_folder, origin_file))
                    os.remove(os.path.join(self.image_folder, annotated_file))
                else:
                    print(f"[SEND] Upload failed for {ts} "
                          f"(origin: {r1.status_code}, annotated: {r2.status_code})")

            except Exception as e:
                print(f"[SEND] Error uploading pair {ts}: {e}")
                
    def send_log_file(self):
        log_path = os.path.join(self.log_folder, 'app.log')

        if not os.path.exists(log_path):
            print("[SEND_LOG] No log file found → skip")
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                print("[SEND_LOG] Empty log file → skip")
                return

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                data = {
                    "host": self.host_ip,
                    "message": line
                }

                resp = requests.post(self.log_endpoint, json=data, timeout=5)
                resp.raise_for_status()

                print(f"[SEND_LOG] Sent: {line}")

            with open(log_path, "w", encoding="utf-8"):
                pass

            print("[SEND_LOG] All logs sent, file cleaned.")

        except Exception as e:
            print(f"[SEND_LOG] Error while sending logs: {e}")



if __name__ == '__main__':
    uploader = ClientUploader('./logs','http://10.0.20.16:8000')
    uploader.send_log_file()
    uploader.send_images()

