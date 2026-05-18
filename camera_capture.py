import cv2
import time

def capture_camera_frame(ip_address, output_file='camera_frame.jpg'):
    """从海康威视摄像头获取画面"""
    rtsp_urls = [
        f"rtsp://admin:12345@{ip_address}:554/Streaming/Channels/101",
        f"rtsp://admin:12345@{ip_address}:554/Streaming/Channels/102",
        f"rtsp://{ip_address}:554/Streaming/Channels/101",
        f"rtsp://{ip_address}/live"
    ]
    
    print(f"正在尝试连接摄像头: {ip_address}")
    
    for i, url in enumerate(rtsp_urls):
        print(f"尝试连接 [{i+1}/{len(rtsp_urls)}]: {url}")
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            cap.release()
            continue
        
        success = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                success = True
                break
            time.sleep(0.5)
        
        if success:
            cv2.imwrite(output_file, frame)
            cap.release()
            print(f"✅ 成功获取画面！已保存到: {output_file}")
            print(f"画面尺寸: {frame.shape[1]} x {frame.shape[0]}")
            return True
        else:
            cap.release()
    
    print("❌ 所有连接方式都失败了")
    return False

if __name__ == "__main__":
    camera_ip = "192.168.1.101"
    capture_camera_frame(camera_ip)