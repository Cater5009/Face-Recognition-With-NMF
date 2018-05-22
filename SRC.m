function rc = SRC(m_img, n_img, V, class_V, Train_num, T, class_T, Test_num)
%V-----------------训练数据集，[m*n, Train_num*15]
%m_img-------------V中图像的维数
%n_img-------------V的样本数
%r-----------------V的秩
%Train_num---------训练样本数
%maxiter-----------最大迭代次数
%T-----------------测试数据集，[m*n, Test_num*15]
%Test_num----------训练样本数

B = [V eye(m_img * n_img)];                                                %扩充矩阵
B = normc(B);                                                              %归一化模
T = normc(T);
e0 = zeros(m_img * n_img, 1);                                              %污染矢量初始化
e = 0.0005 * sqrt(m_img * n_img) * sqrt(1 +...
    2 * sqrt(2)/sqrt(m_img * n_img));

right = 0;
class_recT = zeros(40 * Test_num, 1);
for i = 1 : Test_num * 40
    x0 = B(:, 1 : Train_num * 40)' * T(:,i);                               %最低能量初始化x0
    x0 = [x0; e0];
    xp = l1qc_logbarrier(x0, B, [], T(:, i), e, 1e-4);                     %求解l1最小化问题
    yr = T(:,i) - xp(Train_num * 40 + 1 : end);
    ry = zeros(40, 1);
    for j = 1 : 40                                                         %逐类样本重构
        deltaj = zeros(Train_num * 40, 1);
        deltaj((j - 1) * 8 + 1 : (j - 1) * 8 + 8) = xp((j - 1) * 8 +...
            1 : (j - 1) * 8 + 8);
        ry(j) = norm(yr - B(:, 1 : Train_num * 40) * deltaj)^2;
    end
    [minry index] = sort(ry);                                              %重构误差最小的类作为识别结果
    class_recT(i) = index(1);
    if class_recT(i) == class_T(i)
        right = right + 1;
    end        
end
display(right / (Test_num * 40));
rc = right / (Test_num * 40);
