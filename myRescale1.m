function W = myRescale1(Wv, oksz, ksz)
% Rescales the weights for a smaller kernel size
% Wv - original weights
% oksz - previous kernel size
% ksz - new kernel size

Wt = zeros(oksz,oksz);
cen = (oksz + 1)/2;
rad = (ksz - 1)/2;
Wt(cen-rad:cen+rad,cen-rad:cen+rad) = ones(ksz,ksz);
Wt = Wt(:);
W = Wv(find(Wt == 1),:);

end