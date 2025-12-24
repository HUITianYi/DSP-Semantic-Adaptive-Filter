import cv2
import numpy as np
from rembg import remove, new_session

# 1. 配置路径
input_video_path = 'talking.mp4'
output_mask_path = 'mask_output.mp4'

# 2. 初始化视频流
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"开始处理视频: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

# 3. 初始化视频写入器 (输出黑白Mask视频)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_mask_path, fourcc, fps, (width, height), isColor=True)

# 4. 初始化 rembg session (使用 u2netp 模型，速度快且边缘好)
session = new_session("u2netp")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # --- 核心步骤：AI 分割 ---
    # rembg 不需要复杂的预处理，直接扔进去即可
    # output 是 BGRA 格式，A 通道就是我们要的 Mask
    result = remove(frame, session=session)
    
    # 提取 Alpha 通道 (Mask)
    mask = result[:, :, 3]
    
    # --- 优化：二值化与形态学处理 ---
    # 确保 Mask 是纯黑(0)和纯白(255)，去除边缘半透明噪点
    _, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    # 转换回 BGR 格式以便写入视频 (三个通道都是 Mask)
    mask_bgr = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
    
    out.write(mask_bgr)
    
    if frame_count % 10 == 0:
        print(f"进度: {frame_count}/{total_frames} 帧已生成 Mask...")

cap.release()
out.release()
print(f"Mask 视频生成完毕！已保存为: {output_mask_path}")