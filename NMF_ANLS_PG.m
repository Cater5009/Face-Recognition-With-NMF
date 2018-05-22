function NMF_ANLS_PG( V, m_img, n_img, Train_num, r, maxiter, T, Test_num)
%V-----------------训练数据集，[m*n, Train_num*15]
%m_img-------------V中图像的维数
%n_img-------------V的样本数
%r-----------------V的秩
%Train_num---------训练样本数
%maxiter-----------最大迭代次数
%T-----------------测试数据集，[m*n, Test_num*15]
%Test_num----------训练样本数

%% 训练
W = abs(randn(m_img * n_img, r));                                          %非负初始化
H = abs(randn(r,Train_num * 15));
gradW = W * (H * H') - V * H';                                             %分别计算W梯度和H梯度
gradH = (W' * W) * H - W' * V;    
tol_W = 0.001 * norm([gradW; gradH'],'fro');
tol_H = tol_W;

for iter = 1: maxiter
    [W, gradW, iterW] = nlssubprob(V', H', W', tol_W, 1000); 
    W = W'; 
    gradW = gradW';
    if iterW == 1
        tol_W = 0.1 * tol_W;
    end
    
    [H, gradH, iterH] = nlssubprob(V, W, H, tol_H, 1000);
    if iterH == 1
        tol_H = 0.1 * tol_H;
    end
end
%绘出代价函数和特征
% figure;
% plot([1 : maxiter], J);
figure;
for i = 1 : r
    subplot(5, 8, i);
    im = reshape(W(:, i), m_img, n_img); 
    imagesc(im);colormap('gray');  
end

%% 测试
%迭代，将测试数据表示为W基矢量的线性组合
Ht = abs(randn(r, Test_num * 15));
for iter = 1: maxiter
    Hold = Ht;
    Ht = Hold.* ((W') * T)./((W') * W * Hold + 1e-9);                      %更新H

    norms = sqrt(sum(Ht'.^2));                                             %归一化
    Ht = Ht./(norms'*ones(1,Test_num * 15));
end
VT = W * Ht;                                                               %重构图
%绘出重构图
for i = 1 : Test_num * 15
    if mod(i, 20) == 1
        figure;
        m = 1;
    end
    subplot(4, 5, m);
    im = reshape(VT(:, i), m_img, n_img); 
    imagesc(im);colormap('gray');  
    m = m + 1;
end
%计算匹配率
VT = VT/max(VT(:));
T = T/max(T(:));
e = mean(sum(abs(T - VT))./sum(abs(T)));
display(1 - e);


