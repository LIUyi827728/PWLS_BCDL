function [im,Ddnary]=Generatedictionary(Dataa,Ppsize,DdSize)

Dataa1 = Dataa./ (repmat(sqrt(sum(Dataa.^ 2)), [size(Dataa, 1) 1])+eps);%Dataa��ÿһ���ǿ��γɵ���������
% �ù���Ϊ��Dataa��ÿһ�е�ֵ��Ӻ󿪷����γ�һ����������Ȼ��Ըø�������ƽ�̸��Ƴ�64����size(Dataa,
% 1)��*size(Dataa, 2)��С�ľ���

param.K = DdSize;%�ֵ������
param.lambda = 0.15;%�����ֵ�Ĳ������������ճ���
param.iter = 20;%�ֵ�ѵ���ĵ�������
% load trees;
Ddnary = mexTrainDL(Dataa1, param);%ѵ���ֵ�
% imshow(Ddnary,[]);%��ʾ�ֵ�
% display_dico(Ddnary);

%��ʾ�ֵ�
D=Ddnary;  
K=256;  
figure;  
%����KSVD�������е��ֵ���ʾ����  
im=displayDictionaryElementsAsImage(D, floor(sqrt(K)), floor(size(D,2)/floor(sqrt(K))),Ppsize,Ppsize,0);