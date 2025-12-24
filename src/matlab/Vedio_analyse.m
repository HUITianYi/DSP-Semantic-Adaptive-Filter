%% Digital Signal Processing Project: Differential Denoising & Analysis
% 课题：基于 Mask 的差异化频域去噪及其客观指标分析
% 作者：HTY
% 日期：2025-12-03
clc; clear; close all;

%% 1. 初始化与设置
video_src_path = 'talking.mp4';
video_mask_path = 'mask_output.mp4';

if ~exist(video_src_path, 'file') || ~exist(video_mask_path, 'file')
    error('文件缺失。');
end

v_src = VideoReader(video_src_path);
v_mask = VideoReader(video_mask_path);

H = v_src.Height; W = v_src.Width; fps = v_src.FrameRate;
v_out = VideoWriter('final_analysis_result.avi');
v_out.FrameRate = fps;
open(v_out);

% --- 滤波器预计算 ---
[V, U] = meshgrid(0:W-1, 0:H-1);
idx_u = find(U > H/2); U(idx_u) = U(idx_u) - H;
idx_v = find(V > W/2); V(idx_v) = V(idx_v) - W;
D_matrix = sqrt(U.^2 + V.^2);

D0_person = min(H, W) * 0.15; 
D0_bg     = min(H, W) * 0.05;

H_filter_person = exp(-(D_matrix.^2) ./ (2*(D0_person^2)));
H_filter_bg     = exp(-(D_matrix.^2) ./ (2*(D0_bg^2)));

% --- 数据记录容器 ---
metrics = struct();
metrics.psnr_input_global = []; % 加噪后(输入)的PSNR
metrics.psnr_out_global   = []; % 处理后(输出)的PSNR
metrics.psnr_out_person   = []; % 仅人物区域的PSNR
metrics.psnr_out_bg       = []; % 仅背景区域的PSNR
metrics.ssim_global       = []; % 全局 SSIM

h_fig = figure('Name', 'DSP 处理与实时监测', 'Position', [50, 50, 1200, 600]);

%% 2. 处理循环
frame_idx = 0;
sigma_noise = 0.1; % 噪声强度

fprintf('开始处理并收集数据...\n');

while hasFrame(v_src) && hasFrame(v_mask)
    frame_idx = frame_idx + 1;
    
    % [A] 读取与预处理
    img_gt = im2double(readFrame(v_src)); % Ground Truth (真值)
    img_mask_raw = im2double(readFrame(v_mask));
    
    mask = img_mask_raw(:,:,1);
    mask_bin = mask > 0.5; % 硬阈值 Mask (用于数据统计)
    
    h_blur = fspecial('gaussian', [21 21], 10);
    mask_soft = imfilter(double(mask_bin), h_blur); % 软 Mask (用于图像融合)
    mask_soft_3ch = repmat(mask_soft, [1, 1, 3]);
    
    % [B] 加噪 (Input)
    noise_layer = sigma_noise * randn(H, W, 3);
    img_noisy = img_gt + noise_layer;
    img_noisy(img_noisy > 1) = 1; img_noisy(img_noisy < 0) = 0;
    
    % [C] 频域差异化滤波
    img_p = zeros(size(img_noisy));
    img_b = zeros(size(img_noisy));
    
    for c = 1:3
        F = fft2(img_noisy(:,:,c));
        img_p(:,:,c) = real(ifft2(F .* H_filter_person));
        img_b(:,:,c) = real(ifft2(F .* H_filter_bg));
    end
    
    % [D] 融合 (Output)
    img_out = img_p .* mask_soft_3ch + img_b .* (1 - mask_soft_3ch);
    img_out(img_out > 1) = 1; img_out(img_out < 0) = 0;
    
    % [E] --- 核心：客观指标计算 ---
    
    % 1. 全局 PSNR 计算
    % PSNR = 10 * log10(Peak^2 / MSE)
    p_in_global  = psnr(img_noisy, img_gt);
    p_out_global = psnr(img_out, img_gt);
    
    % 2. 分区 PSNR 计算 (Masked MSE)
    % 提取人物区域的差异
    diff_sq = (img_out - img_gt).^2;
    diff_sq_mean = mean(diff_sq, 3); % RGB通道平均
    
    % 利用逻辑索引提取特定区域的 MSE
    mse_person = mean(diff_sq_mean(mask_bin)); 
    mse_bg     = mean(diff_sq_mean(~mask_bin));
    
    p_out_person = 10 * log10(1 / mse_person);
    p_out_bg     = 10 * log10(1 / mse_bg);
    
    % 3. 全局 SSIM 计算
    s_val = ssim(img_out, img_gt);
    
    % 4. 记录数据
    metrics.psnr_input_global(end+1) = p_in_global;
    metrics.psnr_out_global(end+1)   = p_out_global;
    metrics.psnr_out_person(end+1)   = p_out_person;
    metrics.psnr_out_bg(end+1)       = p_out_bg;
    metrics.ssim_global(end+1)       = s_val;
    
    writeVideo(v_out, im2uint8(img_out));
    
    % [F] 实时可视化
    if mod(frame_idx, 5) == 0
        set(0, 'CurrentFigure', h_fig);
        
        subplot(2,3,1); imshow(img_noisy); title(['Input (PSNR: ' num2str(p_in_global, '%.2f') 'dB)']);
        subplot(2,3,2); imshow(img_out);   title(['Output (PSNR: ' num2str(p_out_global, '%.2f') 'dB)']);
        subplot(2,3,3); imshow(abs(img_out - img_gt) * 5); title('差分图 (误差放大5倍)');
        
        % 实时曲线
        subplot(2,3,4:6); 
        plot(metrics.psnr_out_person, 'r', 'LineWidth', 1.5); hold on;
        plot(metrics.psnr_out_bg, 'b', 'LineWidth', 1.5);
        plot(metrics.psnr_input_global, 'k--', 'LineWidth', 1);
        hold off;
        legend('Person (Output)', 'Background (Output)', 'Noisy Input');
        title('实时分区 PSNR 趋势'); grid on; xlim([1, max(frame_idx, 10)]);
        drawnow;
    end
end
close(v_out);

%% 3. 最终数据分析与绘图 (Analysis Report)

figure('Name', '最终数据分析报告', 'Position', [100, 100, 1000, 600]);

% 图表1: 全局去噪效果对比
subplot(2, 2, 1);
x = 1:length(metrics.psnr_input_global);
area(x, metrics.psnr_out_global, 'FaceColor', 'g', 'FaceAlpha', 0.3); hold on;
plot(x, metrics.psnr_input_global, 'r--', 'LineWidth', 1.5);
title('全局信噪比改善 (Global SNR Improvement)');
ylabel('PSNR (dB)'); xlabel('Frame Index');
legend('Restored Video', 'Noisy Input'); grid on;

% 图表2: 差异化区域对比
subplot(2, 2, 2);
plot(x, metrics.psnr_out_person, 'r-', 'LineWidth', 2); hold on;
plot(x, metrics.psnr_out_bg, 'b-', 'LineWidth', 2);
yline(mean(metrics.psnr_input_global), 'k--', 'Input Level');
title('ROI 分区质量分析 (ROI Analysis)');
legend('Person (Foreground)', 'Background', 'Input Noise Level');
ylabel('PSNR (dB)'); grid on;

% 图表3: SSIM 结构相似性
subplot(2, 2, 3);
plot(x, metrics.ssim_global, 'm-', 'LineWidth', 1.5);
ylim([0, 1]); 
title('SSIM 结构相似性 (越近1越好)');
ylabel('SSIM Index'); grid on;

% 文字统计
avg_gain = mean(metrics.psnr_out_global) - mean(metrics.psnr_input_global);
avg_p_person = mean(metrics.psnr_out_person);
avg_p_bg = mean(metrics.psnr_out_bg);

subplot(2, 2, 4);
axis off;
text(0.1, 0.8, '--- Performance Summary ---', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.6, sprintf('Avg Input PSNR: %.2f dB', mean(metrics.psnr_input_global)), 'FontSize', 12);
text(0.1, 0.5, sprintf('Avg Output PSNR: %.2f dB', mean(metrics.psnr_out_global)), 'FontSize', 12);
text(0.1, 0.4, sprintf('Global Gain: +%.2f dB', avg_gain), 'FontSize', 12, 'Color', 'g', 'FontWeight', 'bold');
text(0.1, 0.2, sprintf('Person Region Quality: %.2f dB', avg_p_person), 'FontSize', 12, 'Color', 'r');
text(0.1, 0.1, sprintf('Background Quality: %.2f dB', avg_p_bg), 'FontSize', 12, 'Color', 'b');

fprintf('分析完成。请查看生成的统计图表。\n');