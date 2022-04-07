classdef Reg_DL < handle
    
    properties
        mMask;  % the mask matrix
        PatSiz; % [rows cols thickness] 1x3 vector patch size
        ImgSiz; % [rows cols thickness] 1x3 vector
        SldDist; % sliding distance
        numBlock;% the number of dictionary
        beta;
        maxatoms;      % maximal number of atoms
        EPSILON; % sparsification error
        mOmega; % dictionary matrix
        mSpa;   % the matrix of sparse code
        rstSpa; % the flag of sparsecode updating
        param;
    end
    
    methods
        function obj = Reg_DL(mask, PatSiz, ImgSiz, SldDist, beta, maxatoms, EPSILON, mOmega, numBlock)
            obj.mMask = mask;
            obj.PatSiz = PatSiz;
            obj.ImgSiz = ImgSiz;
            obj.SldDist = SldDist;
            obj.beta = beta;
            obj.maxatoms = maxatoms;
            obj.EPSILON = EPSILON;
            obj.mOmega = mOmega;
            obj.numBlock=numBlock;
            obj.rstSpa = true;
        end
        
        function cost = penal(obj, A, x, wi, sino)
            % data fidelity
            df = .5 * sum(col(wi) .* (A * x - col(sino)).^2, 'double');
            fprintf('df = %g\n', df);
            x = embed(x, obj.mMask);
            mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
            % sparsity error
            load Patch.mat;
            obj.vIdx=Patch(1,:); 
            mCod = zeros(size(mPat), 'single');
            for k = 1 : obj.numBlock
                tmp = obj.vIdx==k;
                mCod(:,tmp) = obj.mOmega(:,:,k) * obj.mSpa(:,tmp) ;
            end
            
            spa_err = obj.beta *  sum(col(mPat - mCod).^2); clear mCod;
            fprintf('se = %g\n', spa_err);
            spa = nnz(obj.mSpa);
            fprintf('sp = %g\n', spa);
            cost=[]; cost(1)= df; cost(2)= spa_err; cost(3)= spa;
        end
        
        function grad = cgrad(obj, x)
            x = embed(x, obj.mMask);
            mPat = im2colstep(single(x), obj.PatSiz, obj.SldDist); clear x;
            mOmegaT = permute(obj.mOmega,[2 1 3]); 
            G=zeros( size(obj.mOmega,2), size(obj.mOmega,2), obj.numBlock);
            for i=1:obj.numBlock
                G(:,:,i)=mOmegaT(:,:,i)*obj.mOmega(:,:,i);% G = obj.mOmega'* obj.mOmega;
            end
            % update sparse code only at the first inner iteration
            if(obj.rstSpa)
              
               load Patch.mat;
               diff = zeros(size(mPat), 'single');
               for i=1:size(mPat,2)
             
                 obj.mSpa(:,i) = omp2(double(obj.mOmega(:,:,Patch(1,i))' * mPat(:,i)), double(sum(mPat(:,i).*mPat(:,i))), ...
                    G(:,:,Patch(1,i)), obj.EPSILON, 'gammamode','full','maxatoms', obj.maxatoms);
%                 obj.mSpa = omp2(double(obj.mOmega' * mPat), double(sum(mPat.*mPat)), ...
%                     G, obj.EPSILON, 'gammamode','full','maxatoms', obj.maxatoms);
                diff(:,i) = mPat(:,i) - obj.mOmega(:,:,Patch(1,i)) * obj.mSpa(:,i) ;
%                 diff = mPat - obj.mOmega * obj.mSpa;
               end
                obj.rstSpa = false;
                
            else
                
                load Patch.mat;
                diff = zeros(size(mPat), 'single');
                for i=1:size(mPat,2)              
                  diff(:,i) = mPat(:,i) - obj.mOmega(:,:,Patch(1,i)) * obj.mSpa(:,i) ;
                end
%                 diff = mPat - obj.mOmega * obj.mSpa;
            end
            clear mPat;
            grad = 2 * obj.beta .* col2imstep(single(diff), obj.ImgSiz, obj.PatSiz, obj.SldDist);
            grad = grad(obj.mMask);
        end
        
        
        function perc = nextOuterIter(obj)
            % set the flag of updating SparseCode
            obj.rstSpa = true;
            % sparsity check
            perc = nnz(obj.mSpa) / numel(obj.mSpa) * 100;
        end
        
    end
    
end