function rc = PCA(m_img, n_img, V, class_V, Train_num, T, class_T, Test_num, T_orignal)
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

eig_num = 128;                                                             %提取特征维数
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
C = (1/Train_num) * (Vmean' * Vmean);
[Vec, D] = eigs(C, eig_num);                                               %计算eig_num个最大的特征值对应的特征向量
Vec = C' * Vec;                                                            %C 的特征向量
for i = 1 : eig_num                                                        %归一化模
    Vec(:, i) = Vec(:, i)/norm(Vec(:, i));
end

%% 训练
%求特征脸
eigenface = Vmean * Vec;
%绘出特征脸
figure;
for i = 1:eig_num  
    im = eigenface(:,i);   
    im = reshape(im, m_img, n_img);
    subplot(8,16,i);  
    im = imagesc(im);colormap('gray');  
end
suptitle('图1-PCA特征脸');
%将差分训练图投影到特征脸空间
Vproject = Vmean' * eigenface;

%% 测试
%将测试图投影到特征空间
Tproject = Tmean' * eigenface;
A = eigenface \ T;                                                         %测试集到特征脸的投影系数矩阵
T_hat = eigenface * A;                                                     %重构测试样本
AV = eigenface \ V;                                                        %训练集到特征脸的投影系数矩阵
V_hat = eigenface * AV;                                                    %重构训练样本
%绘出重构图
for i = 1 : Test_num * 40
    if mod(i, 20) == 1
        figure;
        m = 1;
    end
    subplot(4, 5, m);
    im = reshape(T_hat(:, i), m_img, n_img); 
    imagesc(im);colormap('gray');  
    m = m + 1;
end
%计算重构准确率
right = 0;
dist = zeros(1, Train_num * 40);
class_recT = zeros(40 * Test_num, 1);
for i = 1 : Test_num * 40
    for j = 1 : Train_num * 40
        dist(j) = norm(A(:, i) - AV(:, j))^2;                              %选取系数的欧氏距离最近的作为识别对象
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