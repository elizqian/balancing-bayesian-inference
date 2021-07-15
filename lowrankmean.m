function [mu, del2] = lowrankmean(y,G,sig_obs,L_pr,r_vals)

d = size(G,2);

mu = zeros(d,length(r_vals));

% low rank quantities
[myV,myS,myW] = svd(G*L_pr/sig_obs);
What = L_pr*myW;
Vhat = myV/sig_obs;
del2 = diag(myS).^2;

for rr = 1:length(r_vals)
    r = r_vals(rr);
    A = What(:,1:r)* diag(sqrt(del2(1:r)) ./ (1 + del2(1:r))) * Vhat(:,1:r)';
    mu(:,rr) = A*y;
end
