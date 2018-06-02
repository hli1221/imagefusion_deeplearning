%load vgg19
net = load('imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

n = 21;
time = zeros(n,1);
for i=2:2
index = i

path1 = ['./IV_images/IR',num2str(index),'.png'];
path2 = ['./IV_images/VIS',num2str(index),'.png'];
fuse_path = ['./fused_infrared/fused',num2str(index),'_VGG_multiLayers.png'];

image1 = imread(path1);
image2 = imread(path2);
image1 = im2double(image1);
image2 = im2double(image2);

tic;
% Highpass filter test image
npd = 16;
fltlmbd = 5;
[I_lrr1, I_saliency1] = lowpass(image1, fltlmbd, npd);
[I_lrr2, I_saliency2] = lowpass(image2, fltlmbd, npd);

%% fuison lrr parts
F_lrr = (I_lrr1+I_lrr2)/2;
figure;imshow(F_lrr);

%% fuison saliency parts use VGG19
disp('VGG19-saliency');
saliency_a = make_3c(I_saliency1);
saliency_b = make_3c(I_saliency2);
saliency_a = single(saliency_a) ; % note: 255 range
saliency_b = single(saliency_b) ; % note: 255 range

res_a = vl_simplenn(net, saliency_a);
res_b = vl_simplenn(net, saliency_b);

%% relu1_1
out_relu1_1_a = res_a(2).x;
out_relu1_1_b = res_b(2).x;
unit_relu1_1 = 1;

l1_featrues_relu1_a = extract_l1_feature(out_relu1_1_a);
l1_featrues_relu1_b = extract_l1_feature(out_relu1_1_b);

[F_saliency_relu1, l1_featrues_relu1_ave_a, l1_featrues_relu1_ave_b] = ...
            fusion_strategy(l1_featrues_relu1_a, l1_featrues_relu1_b, I_saliency1, I_saliency2, unit_relu1_1);

%% relu2_1
out_relu2_1_a = res_a(7).x;
out_relu2_1_b = res_b(7).x;
unit_relu2_1 = 2;

l1_featrues_relu2_a = extract_l1_feature(out_relu2_1_a);
l1_featrues_relu2_b = extract_l1_feature(out_relu2_1_b);

[F_saliency_relu2, l1_featrues_relu2_ave_a, l1_featrues_relu2_ave_b] = ...
            fusion_strategy(l1_featrues_relu2_a, l1_featrues_relu2_b, I_saliency1, I_saliency2, unit_relu2_1);

%% relu3_1
out_relu3_1_a = res_a(12).x;
out_relu3_1_b = res_b(12).x;
unit_relu3_1 = 4;

l1_featrues_relu3_a = extract_l1_feature(out_relu3_1_a);
l1_featrues_relu3_b = extract_l1_feature(out_relu3_1_b);

[F_saliency_relu3, l1_featrues_relu3_ave_a, l1_featrues_relu3_ave_b] = ...
            fusion_strategy(l1_featrues_relu3_a, l1_featrues_relu3_b, I_saliency1, I_saliency2, unit_relu3_1);

%% relu4_1
out_relu4_1_a = res_a(21).x;
out_relu4_1_b = res_b(21).x;
unit_relu4_1 = 8;

l1_featrues_relu4_a = extract_l1_feature(out_relu4_1_a);
l1_featrues_relu4_b = extract_l1_feature(out_relu4_1_b);

[F_saliency_relu4, l1_featrues_relu4_ave_a, l1_featrues_relu4_ave_b] = ...
            fusion_strategy(l1_featrues_relu4_a, l1_featrues_relu4_b, I_saliency1, I_saliency2, unit_relu4_1);

%% fusion strategy
figure;imshow(F_saliency_relu1);
figure;imshow(F_saliency_relu2);
figure;imshow(F_saliency_relu3);
figure;imshow(F_saliency_relu4);

F_saliency = max(F_saliency_relu1, F_saliency_relu2);
F_saliency = max(F_saliency, F_saliency_relu3);
F_saliency = max(F_saliency, F_saliency_relu4);
figure;imshow(F_saliency);

fusion_im = F_lrr + F_saliency;
time(i) = toc;
% figure;imshow(fusion_im);

imwrite(fusion_im,fuse_path,'png');
end


