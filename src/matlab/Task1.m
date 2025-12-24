%% Digital Signal Processing Project
% 主题：全景统一加噪 vs. 基于 Mask 的差异化去噪
% 场景：模拟低光照传感器产生的全图噪点，并测试分区恢复效果
% 作者：HTY
% 日期：2025-12-03
clc; clear; close all;

%% 1. 图像与 Mask 读取预处理
% --- A. 读取原始图像 ---
img_filename = 'PortraitM.jpg'; % 请确保当前目录下有此图
if ~exist(img_filename, 'file')
    error(['找不到文件: ' img_filename]);
end
img_raw = imread(img_filename); 
img_double = im2double(img_raw); 
[rows, cols, channels] = size(img_double);

% --- B. 读取并处理 Mask ---
mask_filename = 'mask.png'; 
if ~exist(mask_filename, 'file')
    % 如果没有Mask，生成一个中心圆形Mask作为测试
    warning('未找到mask.png，生成测试用圆形Mask...');
    [xx, yy] = meshgrid(1:cols, 1:rows);
    mask = double(sqrt((xx-cols/2).^2 + (yy-rows/2).^2) <= min(rows,cols)*0.3);
else
    mask_raw = imread(mask_filename);
    if size(mask_raw, 3) > 1, mask = rgb2gray(mask_raw); else, mask = mask_raw; end
    mask = im2double(mask);
    if size(mask, 1) ~= rows, mask = imresize(mask, [rows, cols]); end
end

% Mask 羽化 (关键：让去噪过渡区不生硬)
h_blur = fspecial('gaussian', [31 31], 15);
mask_soft = imfilter(mask, h_blur);

%% 2. 全景统一加噪 (Global Uniform Noise)
% 模拟传感器热噪声：全图强度一致
sigma_noise = 0.08; % 噪声标准差 (可以调整 0.05 - 0.15)

fprintf('正在施加全景统一噪声 (Sigma = %.2f)...\n', sigma_noise);
noise_matrix = sigma_noise * randn(size(img_double));

img_noisy = img_double + noise_matrix;

% 截断修正
img_noisy(img_noisy > 1) = 1; 
img_noisy(img_noisy < 0) = 0;

figure('Name', '输入信号分析', 'NumberTitle', 'off');
subplot(1,2,1); imshow(img_double); title('原始纯净图像');
subplot(1,2,2); imshow(img_noisy); title(['全图加噪 (Sigma=' num2str(sigma_noise) ')']);

%% 3. 准备频域滤波器
% 动态计算基准尺寸
img_min_dim = min(rows, cols);

% 生成距离矩阵 D
u = 0:(rows-1); v = 0:(cols-1);
idx_u = find(u > rows/2); u(idx_u) = u(idx_u) - rows;
idx_v = find(v > cols/2); v(idx_v) = v(idx_v) - cols;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);

% --- 设计两个不同的滤波器 ---
% 1. 背景滤波器 (H_bg): 截止频率极低，强力去除噪声，哪怕变模糊也无所谓
D0_bg = img_min_dim * 0.06; 
% 2. 人物滤波器 (H_fg): 截止频率较高，为了保留发丝和五官，允许残留少量噪声
D0_fg = img_min_dim * 0.20; 

n = 2; % 巴特沃斯阶数
H_bg = 1 ./ (1 + (D ./ D0_bg).^(2*n));
H_fg = 1 ./ (1 + (D ./ D0_fg).^(2*n));

%% 4. 对比实验：全局滤波 vs 分区滤波
img_global_restore = zeros(size(img_noisy)); % 方法1结果
img_smart_restore  = zeros(size(img_noisy)); % 方法2结果

% 4.1 方法1：传统全局滤波 (使用折中的 D0，比如 0.12)
D0_avg = img_min_dim * 0.12;
H_avg = 1 ./ (1 + (D ./ D0_avg).^(2*n));

fprintf('开始频域处理...\n');
for c = 1:channels
    F = fft2(img_noisy(:, :, c));
    
    % --- A. 传统全局滤波 ---
    G_avg = F .* H_avg;
    img_global_restore(:,:,c) = real(ifft2(G_avg));
    
    % --- B. 智能分区滤波 ---
    % 双路处理
    res_bg = real(ifft2(F .* H_bg)); % 得到一张很糊但在背景很干净的图
    res_fg = real(ifft2(F .* H_fg)); % 得到一张清晰但人物上有少量噪点的图
    
    % 空间融合
    % 公式：Result = 清晰图 * Mask + 模糊图 * (1-Mask)
    img_smart_restore(:,:,c) = res_fg .* mask_soft + res_bg .* (1 - mask_soft);
end

% 截断修正
img_global_restore(img_global_restore > 1) = 1; img_global_restore(img_global_restore < 0) = 0;
img_smart_restore(img_smart_restore > 1) = 1;   img_smart_restore(img_smart_restore < 0) = 0;

%% 5. 结果对比与数据分析
% 计算 PSNR
psnr_in     = psnr(img_noisy, img_double);
psnr_global = psnr(img_global_restore, img_double);
psnr_smart  = psnr(img_smart_restore, img_double);

fprintf('\n=== 实验结果数据 ===\n');
fprintf('1. 原始加噪 PSNR: %.2f dB\n', psnr_in);
fprintf('2. 传统全局 PSNR: %.2f dB (D0=%.1f)\n', psnr_global, D0_avg);
fprintf('3. 智能分区 PSNR: %.2f dB (D0_bg=%.1f, D0_fg=%.1f)\n', psnr_smart, D0_bg, D0_fg);

% 可视化对比
figure('Name', '去噪效果对比', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 500]);

% A. 细节放大区域 (Region of Interest)
roi_r = round(rows/2 - 50 : rows/2 + 50); % 取图像中心100x100区域
roi_c = round(cols/2 - 50 : cols/2 + 50);

subplot(2, 3, 1); imshow(img_noisy); title(['加噪原图 (PSNR: ' num2str(psnr_in,'%.1f') ')']);
subplot(2, 3, 4); imshow(img_noisy(roi_r, roi_c, :)); title('局部细节');

subplot(2, 3, 2); imshow(img_global_restore); title(['传统全局滤波 (PSNR: ' num2str(psnr_global,'%.1f') ')']);
subplot(2, 3, 5); imshow(img_global_restore(roi_r, roi_c, :)); title('局部细节 (变糊了)');

subplot(2, 3, 3); imshow(img_smart_restore); title(['智能分区滤波 (PSNR: ' num2str(psnr_smart,'%.1f') ')']);
subplot(2, 3, 6); imshow(img_smart_restore(roi_r, roi_c, :)); title('局部细节 (保留较好)');

% 绘制 PSNR 柱状图
figure('Name', '性能指标', 'NumberTitle', 'off');
bar([psnr_in, psnr_global, psnr_smart]);
xticklabels({'Noisy Input', 'Global Filter', 'Smart Zone Filter'});
ylabel('PSNR (dB)');
title('去噪算法性能对比');
grid on;