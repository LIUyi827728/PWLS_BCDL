%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear ; clc;close all;
addpath(genpath('../data/2Dxcat'));
addpath(genpath('../toolbox'));
%% setup target geometry and weight
down = 1; % downsample rate
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
ig = image_geom('nx', 512, 'dx', 500/512, 'down', down);
ig.mask = ig.circ > 0;
% A = Gtomo2_dscmex(sg, ig,'nthread', maxNumCompThreads*2);
A = Gtomo2_dscmex(sg, ig,'nthread', jf('ncore')*2-1);
% if neccessary, one could modify maxNumComThreads/jf('ncore') to make full
% use of threads of your machine to accelerate the computation of 
% forward and back projections.

%% load external parameter
I0 = 100;

%load learned dictionary: mOmega
load mOmega.mat;

%load ground truth image: xtrue
xtrue=phantom(512);
xtrue=single(xtrue);
[N,N]=size(xtrue);

dir = ['../data/2Dxcat/tmp/' num2str(I0)];
printm('Loading external sinogram, weight, fbp...');
load([dir '/sino_fan.mat']);
load([dir '/wi.mat']);
% load xfbp as initialization: xrlalm
load([dir '/xfbp.mat']);
xrlalm=xfbp;
% load([dir '/kappa.mat']);
figure name 'xfbp'
imshow(xfbp, []);
%% setup edge-preserving regularizer
ImgSiz =  [ig.nx ig.ny];  % image size
PatSiz =  [8 8];          % patch size
SldDist = [1 1];          % sliding distance

numBlock=6;            % the number of dictionary
nblock = 4;            % Subset Number
nIter = 2;             % I--Inner Iteration
nOuterIter = 40;     % T--Outer Iteration
pixmax = inf;         % Set upper bond for pixel values

printm('Pre-calculating denominator D_A...');
% denom = abs(A)' * col(reshape(sum(abs(A)'), size(wi)) .* wi);
% denom= abs(A)'*(wi(:).*(abs(A)*ones(size(xrla_msk,1),1)));
load([dir '/denom.mat']);

Ab = Gblock(A, nblock); 

% pre-compute D_R
PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);%�ֿ�64*170569��413*413��
KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);%����ת����420*420��С���ص��Ĳ������

for beta =  [7e4]
    for T =  [25]        % maximal number of atoms
        for EPSILON = 55 % sparsification error
            
            fprintf('beta = %.1e, T = %g, EPSILON = %g: \n', beta, T, EPSILON);
            
            D_R = 2 * beta * KK(ig.mask);
            % construct regularizer R(x)
            R = Reg_DL(ig.mask, PatSiz, ImgSiz, SldDist, beta, T, EPSILON, mOmega, numBlock);
            
            info = struct('intensity',I0,'SldDist',SldDist,'beta',beta,'T',T,'EPSILON',EPSILON,...
                'nblock',nblock,'nIter',nIter,'pixmax',pixmax,'transform',mOmega,...
                'xrla',[],'RMSE',[],'SSIM',[],'relE',[],'perc',[],'cost',[]);
            
            xini = xrlalm .* ig.mask;    %initial FBP image
            xrla_msk = xrlalm(ig.mask);
    
            %% Recon
            SqrtPixNum = sqrt(sum(ig.mask(:)>0)); % sqrt(pixel numbers in the mask)
            stop_diff_tol = 1e-3; % HU
            
            % profile on
            for ii=1:nOuterIter
                xold = xrla_msk;
                
                fprintf('Iteration = %d:\n', ii);
                [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(sino,'2d'), reshaper(wi,'2d'),  ...
                    R, denom, D_R, 'pixmax', pixmax, 'chat', 0, 'alpha', 1.999, 'rho', [], 'niter', nIter);
                
                

                xrla = ig.embed(xrla_msk); 
                xnew=xrla;
                x2=xnew;
                x2=reshape(x2,N*N,1);
                minx=min(x2(:));
                maxx=max(x2(:));
                x1=(x2-minx)/(maxx-minx);

                q1=xtrue;
                q1=reshape(q1,N*N,1);
                minx=min(q1(:));
                maxx=max(q1(:));
                q1=(q1-minx)/(maxx-minx);
          %% RMSE
                f=(x1-q1);
                f=f.*f;
                RMSE(ii)=sum(f(:));
                RMSE(ii)=sqrt(RMSE(ii)/(N*N));
 
          %% CORR
                m1 = mean(q1(:));    
                m2 = mean(x1(:));    

                d1=q1-m1;
                d2=x1-m2;
                c1=d1.*d2;
                c1=sum(c1);
                a1=d1.*d1;
                a2=d2.*d2;
                a1=sum(a1);
                a2=sum(a2);
                a3=sqrt(a1.*a2);
                CORR(ii)=c1/a3;  
          %% SNR
                f2=(x1-q1);
                f2=f2.*f2;
                f2=sum(f2);
                a4=a1/f2;
                SNR(ii)=10*log10(a4); 
                fprintf('RMSE = %g, ',RMSE(ii));
                fprintf('CORR= %g, ',CORR(ii));
                fprintf('SNR = %g\n, ',SNR(ii));
                
                
                info.perc(:,ii) = R.nextOuterIter();
                fprintf('perc = %g, ', info.perc(:,ii));
                
                %     info.cost(:,ii) = cost;
                info.relE(:,ii) =  norm(xrla_msk - xold) / SqrtPixNum;
                fprintf('relE = %g\n', info.relE(:,ii));
                if info.relE(:,ii) < stop_diff_tol
                    break
                end
                info.xrla = ig.embed(xrla_msk);
                figure(120), imshow(info.xrla, []); drawnow;
                
            end
            
        end
    end
end

xrla = ig.embed(xrla_msk);
figure;
imshow(xrla, []);



figure;
% 绘制均方误差图
subplot(2,3,1),plot(RMSE,'+r','linewidth',1);
axis([0,40,0.03,0.06]);
grid on;
xlabel('Number of Total Iteration','fontsize',10);
ylabel('RMSE','fontsize',10);

% 绘制相关系数图
subplot(2,3,3),plot(CORR,'+r','linewidth',1);
axis([0,40,0.95,1]);
grid on;
xlabel('Number of Total Iteration','fontsize',10);
ylabel('CORR','fontsize',10);

% 绘制信噪比图形
subplot(2,3,4),plot(SNR,'+r','linewidth',1);
axis([0,40,12,15]);
grid on;
xlabel('Number of Total Iteration','fontsize',10);
ylabel('SNR','fontsize',10);



% xrla=reshape(xrla,N*N,1);
minx=min(xrla(:));
maxx=max(xrla(:));
xrla=(xrla-minx)/(maxx-minx);
% xrla=reshape(xrla,N,N);
b=xrla(300,:);

a=xtrue(300,:);

figure;
% subplot(2,3,5);
plot(a,'b','linewidth',2,'linestyle','--');
axis([0,500,0,1]);
grid on;  % 控制分隔符
% ylim(' auto');
hold on   % 使图形在一个界面上画
plot(b,'r','linewidth',2,'linestyle',':');
xlabel(' Line 300 pixel','fontsize',10);
ylabel('gray value','fontsize',10);
legend('Original image gray scale','Reconstructed image gray scale',1);
