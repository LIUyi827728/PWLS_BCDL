function [im,Ddnary]=Generatedictionary(Dataa,Ppsize,DdSize)

Dataa1 = Dataa./ (repmat(sqrt(sum(Dataa.^ 2)), [size(Dataa, 1) 1])+eps);%Dataa中每一列是块形成的列向量，
% 该过程为：Dataa中每一列的值相加后开方，形成一个行向量，然后对该该行向量平铺复制成64（即size(Dataa,
% 1)）*size(Dataa, 2)大小的矩阵

param.K = DdSize;%字典的列数
param.lambda = 0.15;%序列字典的参数，拉格朗日乘子
param.iter = 20;%字典训练的迭代次数
% load trees;
Ddnary = mexTrainDL(Dataa1, param);%训练字典
% imshow(Ddnary,[]);%显示字典
% display_dico(Ddnary);

%显示字典
D=Ddnary;  
K=256;  
figure;  
%调用KSVD工具箱中的字典显示函数  
im=displayDictionaryElementsAsImage(D, floor(sqrt(K)), floor(size(D,2)/floor(sqrt(K))),Ppsize,Ppsize,0);