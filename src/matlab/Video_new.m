%% Digital Signal Processing Project: Hybrid Python-MATLAB Pipeline
% 课题：基于AI视觉掩膜辅助的差异化频域去噪 (Differential Denoising)
% 视频源：talking.mp4 + mask_output.mp4 (Python生成)
% 作者：HTY
% 日期：2025-12-03
clc; clear; close all;

%% 1. 文件校验与加载
video_src_path = 'talking.mp4';
video_mask_path = 'mask_output.mp4';

if ~exist(video_src_path, 'file') || ~exist(video_mask_path, 'file')
    error('错误：找不到视频文件。请先运行 Python 脚本生成 mask_output.mp4。');
end

v_src = VideoReader(video_src_path);
v_mask = VideoReader(video_mask_path);

% 获取参数
H = v_src.Height;
W = v_src.Width;
fps = v_src.FrameRate;

% 输出视频
v_out = VideoWriter('final_denoise_result.avi');
v_out.FrameRate = fps;
open(v_out);

%% 2. 预计算双通道频域滤波器 (Dual-Filter Pre-computation)
% 我们设计两个不同的滤波器：
% H_high: 截止频率高，保留细节 (用于人物)
% H_low:  截止频率低，强力模糊 (用于背景)

% 生成距离矩阵
u = 0:(H-1); v = 0:(W-1);
idx_u = find(u > H/2); u(idx_u) = u(idx_u) - H;
idx_v = find(v > W/2); v(idx_v) = v(idx_v) - W;
[V, U] = meshgrid(v, u);
D_matrix = sqrt(U.^2 + V.^2);

% --- 关键参数设置 ---
D0_person = min(H, W) * 0.15;  % 人物：截止频率较高 (保留五官)
D0_bg     = min(H, W) * 0.05;  % 背景：截止频率极低 (强力抹平噪声)

% 生成两个高斯低通滤波器
H_filter_person = exp(-(D_matrix.^2) ./ (2*(D0_person^2)));
H_filter_bg     = exp(-(D_matrix.^2) ./ (2*(D0_bg^2)));

%% 3. 双流处理循环 (Dual-Stream Processing)
h_fig = figure('Name', 'DSP 差异化去噪流水线', 'Position', [100, 100, 1200, 350]);
frame_idx = 0;
fprintf('开始 DSP 差异化去噪处理...\n');

while hasFrame(v_src) && hasFrame(v_mask)
    frame_idx = frame_idx + 1;
    
    % --- A. 读取双视频流 ---
    img_raw = im2double(readFrame(v_src));
    img_mask_rgb = im2double(readFrame(v_mask));
    
    % --- B. 处理 Mask ---
    mask = img_mask_rgb(:,:,1); 
    mask(mask > 0.5) = 1; mask(mask <= 0.5) = 0; % 二值化
    
    % 羽化 Mask (融合的关键)
    h_blur = fspecial('gaussian', [21 21], 10); % 稍微加大羽化范围
    mask_soft = imfilter(mask, h_blur);
    mask_soft_3ch = repmat(mask_soft, [1, 1, 3]); % 扩展为3通道
    
    % --- C. 全局加噪 (Global Noise Injection) ---
    % 模拟恶劣环境，对全图施加统一的大噪声
    sigma_noise = 0.1; % 噪声强度 (0.05~0.15 之间调整)
    noise_layer = sigma_noise * randn(H, W, 3);
    img_noisy = img_raw + noise_layer;
    
    % 截断修正 (防止超出 [0,1])
    img_noisy(img_noisy > 1) = 1; img_noisy(img_noisy < 0) = 0;
    
    % --- D. 双通道 FFT 滤波 (Dual-Channel Filtering) ---
    img_restored_person = zeros(size(img_noisy));
    img_restored_bg     = zeros(size(img_noisy));
    
    for c = 1:3
        % 1. 变换到频域
        F = fft2(img_noisy(:,:,c));
        
        % 2. 应用两个不同的滤波器
        G_person = F .* H_filter_person;
        G_bg     = F .* H_filter_bg;
        
        % 3. 逆变换回空域
        img_restored_person(:,:,c) = real(ifft2(G_person));
        img_restored_bg(:,:,c)     = real(ifft2(G_bg));
    end
    
    % --- E. 空间域融合 (Spatial Blending) ---
    % 公式：Result = Mask * Person_Layer + (1-Mask) * Background_Layer
    % 解释：Mask为白色(1)的地方使用清晰图，Mask为黑色(0)的地方使用模糊图
    img_final = img_restored_person .* mask_soft_3ch + ...
                img_restored_bg .* (1 - mask_soft_3ch);
            
    % 再次截断修正
    img_final(img_final > 1) = 1; img_final(img_final < 0) = 0;
    
    % --- F. 写入结果 ---
    writeVideo(v_out, im2uint8(img_final));
    
    % --- G. 实时可视化 ---
    if mod(frame_idx, 5) == 0
        set(0, 'CurrentFigure', h_fig);
        
        % 显示加噪后的图
        subplot(1,4,1); imshow(img_noisy); 
        title(['全局加噪 input (\sigma=' num2str(sigma_noise) ')']);
        
        % 显示 Mask
        subplot(1,4,2); imshow(mask_soft); 
        title('AI Mask (羽化权重)');
        
        % 显示"仅人物"处理结果 (中间变量，用于调试)
        subplot(1,4,3); imshow(img_restored_person); 
        title('高频保留 (Detail)');
        
        % 显示最终融合结果
        subplot(1,4,4); imshow(img_final); 
        title('差异化去噪结果 (Output)');
        drawnow;
    end
end

close(v_out);
fprintf('处理完成！最终视频已保存: final_denoise_result.avi\n');