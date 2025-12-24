# gen_mask.py
import cv2
import numpy as np
from rembg import remove

# 1. 读取图像
input_path = 'PortraitM.jpg'
output_mask_path = 'mask.png'

print(f"正在处理图像: {input_path} ...")
img = cv2.imread(input_path)

# 2. 使用 rembg 移除背景 (生成带Alpha通道的图)
# rembg 会自动识别人物
img_rgba = remove(img)

# 3. 提取 Alpha 通道作为 Mask
# Alpha通道中，0是背景，255是前景
mask = img_rgba[:, :, 3]

# 4. 保存 Mask
cv2.imwrite(output_mask_path, mask)
print(f"Mask 已生成并保存为: {output_mask_path}")