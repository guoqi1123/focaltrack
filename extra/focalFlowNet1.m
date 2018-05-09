function [z_est, u, ATA] = focalFlowNet1(I,netParam,P)
% the focalFlowNet is a network that can be specifically used to compute
% depth from images using focal flow theorem
% Author: Qi Guo, Harvard Universty, All Rights Reserved
% Email: qguo@seas.harvard.edu
%% Input and output instructions
% INPUT:
% I: three mxnx(2k+1) images, in 0-1 double format
% netParam: a structure that contains all the parameters
% OUTPUT:
% z_est: estimated depth

assert(mod(size(I,3),2) == 1,'There must be 2k+1 images!');
k = (size(I,3) - 1) / 2;

% the filter size of fx is filter_y * 2 + 1:filter_x * 2 + 1
assert(mod(size(netParam.diffFilter.fx,2),2) == 1,'The length of the filter must be odd!');
filter_x = (size(netParam.diffFilter.fx,2) - 1)/2;
filter_y = (size(netParam.diffFilter.fx,1) - 1)/2;
assert(filter_x >= filter_y,'The x directional filter must be fat!');
m = size(I,1);
n = size(I,2);
%% Layer1: Convolution with differential filters
Ix = conv2(I(:,:,k+1),netParam.diffFilter.fx,'same') ./ netParam.diffFilter.dunits;
Ixx = conv2(Ix,netParam.diffFilter.fx,'same') ./ netParam.diffFilter.dunits;
Iy = conv2(I(:,:,k+1),netParam.diffFilter.fy,'same') ./ netParam.diffFilter.dunits;
Iyy = conv2(Iy,netParam.diffFilter.fy,'same') ./ netParam.diffFilter.dunits;

Ft = repmat(reshape(netParam.diffFilter.ft ./ netParam.diffFilter.tunits,1,1,2 * k + 1),m,n,1);
It = sum(I .* Ft,3);

% remain only the valid part
Ix = Ix(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y));
Ixx = Ixx(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y));
Iy = Iy(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y));
Iyy = Iyy(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y));

[xx,yy] = meshgrid(1:size(Ix,2),1:size(Ix,1));
xx = xx - (size(Ix,2) + 1)/2;
yy = yy - (size(Ix,1) + 1)/2;

xIx = xx .* Ix;
yIy = yy .* Iy;

It = It(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y));

%% Layer2: Nonlinearity
N11 = Ix.^2;
N12 = Ix .* Iy;
N13 = Ix .* (xIx + yIy);
N14 = Ix .* (Ixx + Iyy);

N22 = Iy.^2;
N23 = Iy .* (xIx + yIy);
N24 = Iy .* (Ixx + Iyy);

N33 = (xIx + yIy) .* (xIx + yIy);
N34 = (xIx + yIy) .* (Ixx + Iyy);

N44 = (Ixx + Iyy) .* (Ixx + Iyy);

T1 = Ix .* It;
T2 = Iy .* It;
T3 = (xIx + yIy) .* It;
T4 = (Ixx + Iyy) .* It;

%% Layer3: Apply the Window Function
A11 = conv2(N11, netParam.w, 'valid');
A12 = conv2(N12, netParam.w, 'valid');
A13 = conv2(N13, netParam.w, 'valid');
A14 = conv2(N14, netParam.w, 'valid');

A21 = A12;
A22 = conv2(N22, netParam.w, 'valid');
A23 = conv2(N23, netParam.w, 'valid');
A24 = conv2(N24, netParam.w, 'valid');

A31 = A13;
A32 = A23;
A33 = conv2(N33, netParam.w, 'valid');
A34 = conv2(N34, netParam.w, 'valid');

A41 = A14;
A42 = A24;
A43 = A34;
A44 = conv2(N44, netParam.w, 'valid');

B1 = conv2(T1, netParam.w, 'valid'); 
B2 = conv2(T2, netParam.w, 'valid'); 
B3 = conv2(T3, netParam.w, 'valid'); 
B4 = conv2(T4, netParam.w, 'valid'); 

%% Layer4: Sample only one point from Axx
ATA = zeros(4,4);
ATb = zeros(4,1);

ptc = 1;
ptr = 1;

ATA(1,1) = A11(ptc,ptr);
ATA(1,2) = A12(ptc,ptr);
ATA(1,3) = A13(ptc,ptr);
ATA(1,4) = A14(ptc,ptr);

ATA(2,1) = A21(ptc,ptr);
ATA(2,2) = A22(ptc,ptr);
ATA(2,3) = A23(ptc,ptr);
ATA(2,4) = A24(ptc,ptr);

ATA(3,1) = A31(ptc,ptr);
ATA(3,2) = A32(ptc,ptr);
ATA(3,3) = A33(ptc,ptr);
ATA(3,4) = A34(ptc,ptr);

ATA(4,1) = A41(ptc,ptr);
ATA(4,2) = A42(ptc,ptr);
ATA(4,3) = A43(ptc,ptr);
ATA(4,4) = A44(ptc,ptr);

ATb(1) = -B1(ptc,ptr);
ATb(2) = -B2(ptc,ptr);
ATb(3) = -B3(ptc,ptr);
ATb(4) = -B4(ptc,ptr);

%% Layer5: compute u
u = ATA \ ATb;

%% Layer6: compute 
mu_f = 1 ./ (1 ./ netParam.camParam.f - 1 ./ netParam.camParam.mu_s);
z_est = ((netParam.camParam.mu_s / netParam.camParam.pixSize)^2 * ...
    (netParam.camParam.eqSigma(1,1) / netParam.camParam.pixSize)^2 * ...
    (mu_f / netParam.camParam.pixSize)) .* u(3) ./ ...
    ((netParam.camParam.mu_s / netParam.camParam.pixSize)^2 * ...
    (netParam.camParam.eqSigma(1,1) / netParam.camParam.pixSize)^2 .* u(3) - ...
    (mu_f / netParam.camParam.pixSize)^2 * u(4)) * netParam.camParam.pixSize;

%% Optional: we compute R for further investigation
% R = - u(1) * Ix - u(2) * Iy - u(3) * (xIx + yIy) - It;
% subplot(1,3,1)
% imshow(P(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),k+1),[]);
% title('Pinhole Image');
% subplot(1,3,2);
% imshow(I(1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),1+2*(filter_x + filter_y):end-2*(filter_x + filter_y),k+1),[]);
% title('Captured Image');
% subplot(1,3,3);
% imshow(R,[]);
% title('Residual R');
end
















