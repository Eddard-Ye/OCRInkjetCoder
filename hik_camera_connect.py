# -*- coding: utf-8 -*-
import sys
import os
import time
import ctypes
from ctypes import *

import numpy as np
import cv2

# 海康SDK路径
HIK_SDK_PATH = r"D:\海康威视\MVS\Development\Samples\Python"
SDK_PATH = os.path.join(HIK_SDK_PATH, "MvImport")
sys.path.append(SDK_PATH)

from MvCameraControl_class import *
from CameraParams_header import *

class HikCamera:
    def __init__(self):
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.st_device_list = None
        self.n_connection_num = 0
        self.b_open_device = False
        self.b_start_grabbing = False
        self.handle = None

    def enum_devices(self, device_type=MV_GIGE_DEVICE):
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        ret = self.cam.MV_CC_EnumDevices(device_type, self.device_list)
        if ret != 0:
            print(f"枚举设备失败，错误码: 0x{ret:x}")
            return False

        print(f"枚举到 {self.device_list.nDeviceNum} 个设备")
        return True

    @staticmethod
    def decoding_char(ctypes_char_array):
        byte_str = memoryview(ctypes_char_array).tobytes()
        null_index = byte_str.find(b'\x00')
        if null_index != -1:
            byte_str = byte_str[:null_index]
        for encoding in ['gbk', 'utf-8', 'latin-1']:
            try:
                return byte_str.decode(encoding)
            except UnicodeDecodeError:
                continue
        return byte_str.decode('latin-1', errors='ignore')

    @staticmethod
    def ip_int_to_str(ip_int):
        nip1 = ((ip_int & 0xff000000) >> 24)
        nip2 = ((ip_int & 0x00ff0000) >> 16)
        nip3 = ((ip_int & 0x0000ff00) >> 8)
        nip4 = (ip_int & 0x000000ff)
        return f"{nip1}.{nip2}.{nip3}.{nip4}"

    def get_device_info(self, index=0):
        if index >= self.device_list.nDeviceNum:
            print(f"设备索引 {index} 超出范围")
            return None

        stDevInfo = cast(self.device_list.pDeviceInfo[index],
                        POINTER(MV_CC_DEVICE_INFO)).contents

        if stDevInfo.nTLayerType == MV_GIGE_DEVICE:
            print(f"\n设备[{index}]: GigE相机")
            stGigEInfo = stDevInfo.SpecialInfo.stGigEInfo
            print(f"  用户定义名称: {self.decoding_char(stGigEInfo.chUserDefinedName)}")
            print(f"  型号: {self.decoding_char(stGigEInfo.chModelName)}")
            print(f"  固件版本: {self.decoding_char(stGigEInfo.chDeviceVersion)}")
            print(f"  IP地址: {self.ip_int_to_str(stGigEInfo.nCurrentIp)}")
            print(f"  序列号: {self.decoding_char(stGigEInfo.chSerialNumber)}")
        elif stDevInfo.nTLayerType == MV_USB_DEVICE:
            print(f"\n设备[{index}]: USB相机")
            stUsbInfo = stDevInfo.SpecialInfo.stUsb3VInfo
            print(f"  用户定义名称: {self.decoding_char(stUsbInfo.chUserDefinedName)}")
            print(f"  型号: {self.decoding_char(stUsbInfo.chModelName)}")
        elif stDevInfo.nTLayerType == MV_CAMERALINK_DEVICE:
            print(f"\n设备[{index}]: CameraLink相机")

        return stDevInfo

    def open_device(self, connection_num=0):
        if self.b_open_device:
            print("设备已经打开")
            return True

        stDevInfo = self.get_device_info(connection_num)
        if stDevInfo is None:
            return False

        self.n_connection_num = connection_num

        ret = self.cam.MV_CC_CreateHandle(stDevInfo)
        if ret != 0:
            print(f"创建设备句柄失败，错误码: 0x{ret:x}")
            return False

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
        if ret != 0:
            print(f"打开设备失败，错误码: 0x{ret:x}")
            self.cam.MV_CC_DestroyHandle()
            return False

        self.b_open_device = True
        print("设备打开成功！")

        if stDevInfo.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print(f"设置数据包大小失败，错误码: 0x{ret:x}")

        return True

    def close_device(self):
        if self.b_start_grabbing:
            self.stop_grabbing()

        if self.b_open_device:
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                print(f"关闭设备失败，错误码: 0x{ret:x}")
                return False

            self.cam.MV_CC_DestroyHandle()
            self.b_open_device = False
            print("设备已关闭")
        return True

    def start_grabbing(self, buffer_num=3):
        if not self.b_open_device:
            print("请先打开设备")
            return False

        if self.b_start_grabbing:
            print("已经在采集")
            return True

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始采集失败，错误码: 0x{ret:x}")
            return False

        self.b_start_grabbing = True
        print("开始采集成功！")
        return True

    def stop_grabbing(self):
        if not self.b_start_grabbing:
            return True

        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print(f"停止采集失败，错误码: 0x{ret:x}")
            return False

        self.b_start_grabbing = False
        print("停止采集成功")
        return True

    def get_one_frame(self, timeout=1000):
        stFrameInfo = MV_FRAME_OUT()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        print(f"正在从相机获取图像 (超时: {timeout}ms)...")
        ret = self.cam.MV_CC_GetImageBuffer(stFrameInfo, timeout)
        if ret != 0:
            print(f"获取图像失败，错误码: 0x{ret:x}")
            return None

        if stFrameInfo.pBufAddr is None:
            print("图像缓冲区地址为空")
            return None

        width = stFrameInfo.stFrameInfo.nWidth
        height = stFrameInfo.stFrameInfo.nHeight
        pixel_type = stFrameInfo.stFrameInfo.enPixelType
        img_size = stFrameInfo.stFrameInfo.nFrameLen

        print(f"图像信息: {width}x{height}, 像素格式: {pixel_type}, 大小: {img_size} bytes")

        frame_data = (c_ubyte * img_size)()
        memmove(frame_data, stFrameInfo.pBufAddr, img_size)

        self.cam.MV_CC_FreeImageBuffer(stFrameInfo)

        if pixel_type == PixelType_Gvsp_Mono8:
            img = np.array(frame_data, dtype=np.uint8).reshape((height, width))
        elif pixel_type in [PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGR8,
                           PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
            img = np.array(frame_data, dtype=np.uint8).reshape((height, width))
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            img = np.array(frame_data, dtype=np.uint8).reshape((height, width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            print(f"不支持的像素格式: {pixel_type}")
            return None

        return img

    def capture_frame(self, output_path='camera_frame.jpg', timeout=1000):
        if not self.b_start_grabbing:
            print("请先开始采集")
            return False

        img = self.get_one_frame(timeout)
        if img is None:
            print("获取图像失败")
            return False

        output_path = os.path.abspath(output_path)
        print(f"尝试保存图像到: {output_path}")
        print(f"图像类型: {type(img)}, 形状: {img.shape if hasattr(img, 'shape') else 'N/A'}")

        success = cv2.imwrite(output_path, img)
        if success:
            print(f"图像已保存到: {output_path}")
        else:
            print(f"保存图像失败! imwrite返回: {success}")
            if os.path.exists(output_path):
                print(f"文件已存在: {output_path}")
                os.remove(output_path)
                print("已删除旧文件，重试保存...")
                success = cv2.imwrite(output_path, img)
                if success:
                    print(f"重试成功！图像已保存到: {output_path}")
                else:
                    print(f"重试仍然失败!")
        return success


def main():
    camera = HikCamera()

    print("=" * 50)
    print("海康工业相机连接程序")
    print("=" * 50)

    print("\n[1] 枚举GigE设备...")
    if not camera.enum_devices(MV_GIGE_DEVICE):
        print("枚举设备失败")
        return

    if camera.device_list.nDeviceNum == 0:
        print("未找到任何设备，请检查相机连接")
        return

    camera.get_device_info(0)

    print("\n[2] 连接设备...")
    if not camera.open_device(0):
        print("连接设备失败")
        return

    print("\n[3] 开始采集...")
    if not camera.start_grabbing():
        camera.close_device()
        return

    print("\n[4] 捕获图像...")
    output_file = os.path.join(os.path.expanduser("~"), 'camera_frame.jpg')
    success = camera.capture_frame(output_file, timeout=3000)

    if success:
        print("\n" + "=" * 50)
        print("✅ 成功！图像已保存")
        print("=" * 50)

        img = cv2.imread(output_file)
        if img is not None:
            print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}")
    else:
        print("\n❌ 捕获失败")

    print("\n[5] 关闭设备...")
    camera.stop_grabbing()
    camera.close_device()

    print("\n程序结束")


if __name__ == "__main__":
    main()
