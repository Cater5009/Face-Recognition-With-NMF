function rc = NMF_mse( V, class_V, m_img, n_img, Train_num, r, maxiter, T, class_T, Test_num, T_orignal)
%V-----------------训练数据集，[m*n, Train_num*15]
%class_V-----------训练样本对应的分类
%m_img-------------V中图像的维数
%n_img-------------V的样本数
%r-----------------V的秩
%Train_num---------训练样本数
%maxiter-----------最大迭代次数
%T-----------------测试数据集，[m*n, Test_num*15]
%class_T-----------测试样本对应的分类
%Test_num----------训练样本数

%% 训练
J = zeros(maxiter, 1);
V = V / max(V(:));
W = abs(randn(m_img * n_img, r));                                          %非负初始化
H = abs(randn(r, Train_num * 40));
J(1) = 0.5 * sum(sum((V - W * H).^2));                                     %代价函数为欧氏距离

for iter = 1: maxiter
    Wold = W;
    Hold = H;
    H = Hold.* ((Wold') * V)./((Wold') * Wold * Hold + 1e-9);              %更新W和H
    W = Wold.* (V * (H'))./(Wold * H * (H') + 1e-9);

    norms = sqrt(sum(H'.^2));                                              %归一化
    H = H./(norms'*ones(1, Train_num * 40));
    W = W.*(ones(m_img * n_img, 1)*norms);
    
    J(iter) = 0.5 * sum(sum(( V - W * H).^2));                             %更新代价函数
end
%绘出代价函数和特征
figure;
plot([1 : maxiter], J);
% figure;
% for i = 1 : r
%     subplot(8, 16, i);
%     im = reshape(W(:, i), m_img, n_img); 
%     imagesc(im);colormap('gray');  
% end

%% 测试
%迭代，将测试数据表示为W基矢量的线性组合
Ht = abs(randn(r, Test_num * 40));
for iter = 1: maxiter
    Hold = Ht;
    Ht = Hold.* ((W') * T)./((W') * W * Hold + 1e-9);                      %更新H

    norms = sqrt(sum(Ht'.^2));                                             %归一化
    Ht = Ht./(norms'*ones(1,Test_num * 40));
end
rec_V = W * H;
rec_T = W * Ht;                                                            %重构图
%绘出重构图
for i = 1 : Test_num * 40
    if mod(i, 20) == 1
        figure;
        m = 1;
    end
    subplot(4, 5, m);
    im = reshape(rec_T(:, i), m_img, n_img); 
    imagesc(im);colormap('gray');  
    m = m + 1;
end
%计算匹配率
right = 0;
dist = zeros(1, Train_num * 40);
class_recT = zeros(40 * Test_num, 1);
for i = 1 : Test_num * 40
    for j = 1 : Train_num * 40
        dist(j) = norm(Ht(:, i) - H(:, j))^2;                              %选取系数的欧氏距离最近的作为识别对象
    end
    [mindist index] = sort(dist);
    class_recT(i) = class_V(index(1));
end
for i = 1 : Test_num * 40                                                  %统计识别率
    if class_recT(i) == class_T(i)
        right = right + 1;
    end
end
display(right / (Test_num * 40));
rc = right / (Test_num * 40);

