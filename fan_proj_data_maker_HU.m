%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear; close all; 
%% generate noisy sinogram and statistical weighting
ff = 50;

down = 1; % downsample rate
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
ig_big = image_geom('nx', 512, 'dx', 500/512);
dir = ['./data/2Dxcat/tmp/' num2str(ff)];

xtrue=phantom(512);
xtrue=single(xtrue);
[N,N]=size(xtrue);
figure;
imshow(xtrue,[]);
num=N*N;
q=reshape(xtrue,num,1);
ggg=1e3;
q=reshape(q,num,1);
min1=min(q);
max1=max(q);
q=ggg*(q-min1)/(max1-min1);

Abig = Gtomo2_dscmex(sg, ig_big);  clear ig_big;
sino_true = Abig * q; clear Abig xtrue_hi;
sino_true=reshape(sino_true,[888 984]);
figure name 'true sinogram'
imshow(sino_true, []);
fprintf('adding noise...\n');
ff=50;
uu=22000;
V=ff*exp(sino_true/uu);
sino=normrnd(sino_true,(V));
ind=find(sino<0);
sino(ind)=0;
sino=reshape(sino,[888 984]);
figure name 'Noisy sinogram'
imshow(sino, []);
wi=1./V;
 
% save([dir '/wi.mat'], 'wi');    %wi -V
% save([dir '/sino_fan.mat'], 'sino'); 

%% setup target geometry and fbp
ig = image_geom('nx', 512, 'dx', 500/512, 'down', down);

ig.mask = ig.circ > 0;
A = Gtomo2_dscmex(sg, ig);
fprintf('fbp...\n');

tmp = fbp2(sg, ig);
xfbp = fbp2(sino, tmp, 'window', 'hanning,0.4'); clear tmp;
xfbp = max(xfbp, 0);
% save([dir '/xfbp.mat'], 'xfbp'); 
figure name 'xfbp'
imshow(xfbp, []);

%% setup kappa
fprintf('calculating kappa...\n');
kappa = sqrt( div0(A' * wi, A' * ones(size(wi))) );
save([dir '/kappa.mat'], 'kappa');

%% setup diag{A'WA1}
printm('Pre-calculating denominator D_A...');
denom = A' * col(reshape(sum(A'), size(wi)) .* wi); 
save([dir '/denom.mat'], 'denom');
