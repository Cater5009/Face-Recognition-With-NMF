function NMF_KL( V, m_img, n_img, Train_num, r, maxiter, T, Test_num)
%V-----------------训练数据集，[m*n, Train_num*15]
%m_img-------------V中图像的维数
%n_img-------------V的样本数
%r-----------------V的秩
%Train_num---------训练样本数
%maxiter-----------最大迭代次数
%T-----------------测试数据集，[m*n, Test_num*15]
%Test_num----------训练样本数

%% 训练
J = zeros(maxiter, 1);
V = V/max(V(:));                                                           %归一化
V(V == 0) = 2;                                                             %替换0值，避免代价函数发散
V(V == 2) = min(min(V)) * 0.01;
W = abs(randn(m_img * n_img, r));                                          %非负初始化
H = abs(randn(r,Train_num * 15));
J(1) = sum(sum((V.* log(V./(W * H))) - V + W * H));                        %代价函数为KL散度

for iter = 1: maxiter
    Wold = W;
    Hold = H;
    W = Wold.*((V./(Wold * Hold + 1e-9)) * Hold')./(ones(m_img * n_img,1) * sum(Hold'));%更新W和H
    H = Hold.*(W' * (V./(W * Hold + 1e-9)))./(sum(W)' * ones(1, Train_num * 15));

    norms = sqrt(sum(H'.^2));                                              %归一化
    H = H./(norms'*ones(1,Train_num * 15));
    W = W.*(ones(m_img * n_img,1)*norms);
    
    J(iter) = sum(sum((V.* log(V./(W * H))) - V + W * H));                 %更新代价函数
end
%绘出代价函数和特征
figure;
plot([1 : maxiter], J);
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
    Ht = Hold.* ((W') * T)./((W') * W * Hold + 1e-9);                      %更新W和H

    norms = sqrt(sum(Ht'.^2));                                             %归一化
    Ht = Ht./(norms'*ones(1,Test_num * 15));
end
VT = W * Ht;
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
