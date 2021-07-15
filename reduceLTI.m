function [Ar,Br,Cr,Sr,Tr,hankel] = reduceLTI(r,A,B,C,L,R)


% SVD-based transform
[U,S,V] = svd(L(:,1:r)'*R(:,1:r));
siginvsqrt = S^-0.5;
Sr = (siginvsqrt*U'*L(:,1:r)')';
Tr = R(:,1:r)*V*siginvsqrt;
hankel = diag(S);

% GEV-based transform
% d = size(Q,1);
% Pinv = inv(P);
% [V,D] = eig(Q,Pinv);
% [~,a] = sort(diag(D),'descend');
% V = V(:,a);
% for i = 1:d     % normalize
%     V(:,i) = V(:,i) / sqrt(V(:,i)'*Pinv*V(:,i));
% end
% hankel = sqrt(diag(D(a,a)));
% Sig = diag(hankel);
% 
% siginvsqrt = Sig^-0.5;
% Tinv = siginvsqrt*V';
% T = inv(Tinv);
% Tr = T(:,1:r);
% Sr = Tinv(1:r,:)';

Ar = Sr'*A*Tr;
Br = Sr'*B;
Cr = C*Tr;

