function rc = PCA_SRC( V, class_V, Train_num, T, class_T, Test_num)
%V-----------------训练数据集，[m*n, Train_num*15]
%m_img-------------V中图像的维数
%n_img-------------V的样本数
%r-----------------V的秩
%Train_num---------训练样本数
%maxiter-----------最大迭代次数
%T-----------------测试数据集，[m*n, Test_num*15]
%Test_num----------训练样本数

eig_num = 64;                                                              %提取特征维数
%% 获取差分图像，计算协方差，并求取特征值
Vmean = V;
Tmean = T;
%差分
mean_V = mean(V,2);
mean_T = mean(T,2);
for i = 1:Train_num * 40
    Vmean(:,i) = V(:,i) - mean_V;
end
Vmean = double(Vmean);                                                     %去中心化
for i = 1:Test_num * 40
    Tmean(:,i) = T(:,i) - mean_T;   
end
Tmean = double(Tmean);
%协方差、特征向量
C = (1/Train_num) * (Vmean * Vmean');
[Vec, D] = eigs(C, eig_num);                                               %计算eig_num个最大的特征值对应的特征向量
Vec = C' * Vec;                                                            %C 的特征向量
for i = 1 : eig_num
    Vec(:, i) = Vec(:, i)/norm(Vec(:, i));
end

%% 训练
%求特征脸
eigenface = (Vmean' * Vec)';                                               %训练集到特征空间的投影（特征脸）
eigenfaceT = (Tmean' * Vec)';                                              %测试集到特征空间的投影
eigenface = normc(eigenface);                                              %归一化模
eigenfaceT = normc(eigenfaceT);
e = 0.0005 * sqrt(eig_num) * sqrt(1 + 2 * sqrt(2)/sqrt(eig_num));          %定义l1范数最小化的重构误差阈值

right = 0;
class_recT = zeros(40 * Test_num, 1);
for i = 1 : Test_num * 40
    x0 = eigenface' * eigenfaceT(:,i);                                     %最低能量初始化x0
    xp = l1qc_logbarrier(x0, eigenface, [], eigenfaceT(:, i), e, 1e-4);    %求解l1最小化问题
    ry = zeros(40, 1);
    for j = 1 : 40                                                         %逐类样本重构
        deltaj = zeros(Train_num * 40, 1);
        deltaj((j - 1) * 8 + 1 : (j - 1) * 8 + 8) = xp((j - 1) * 8 +...
            1 : (j - 1) * 8 + 8);
        ry(j) = norm(eigenfaceT(:,i) - eigenface * deltaj)^2;
    end
    [minry index] = sort(ry);                                              %重构误差最小的类作为识别结果
    class_recT(i) = index(1);
    if class_recT(i) == class_T(i)
        right = right + 1;
    end        
end
display(right / (Test_num * 40));
rc = right / (Test_num * 40);



