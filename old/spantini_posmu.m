function [mu1,mu2] = spantini_posmu(y,G,sig_obs,L_pr,Gpos,sp_svd,r_vals)

% What = L_pr*sp_svd.W;
% Vhat = sp_svd.V/sig_obs;
% del2 = sp_svd.S.^2;

d = size(L_pr,1);
rhs = G'*(y/sig_obs^2);

[mu1,mu2] = deal(zeros(d,length(r_vals)));
for rr = 1:length(r_vals)
%     r = r_vals(rr);
%     A = What(:,1:r)* diag(sqrt(del2(1:r)) ./ (1 + del2(1:r))) * Vhat(:,1:r)';
    mu2(:,rr) = Gpos(:,:,rr)*rhs;
%     mu1(:,rr) = A*y;
end
