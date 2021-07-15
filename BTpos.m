function [Gpos, hankel,info] = BTpos(A,B,C,sig_obs,L,R,r_vals,obs_times)
if nargin == 7
    method = 'lyap';
else 
    method = 'direct';
end

d = size(A,1);
[Ar,~,Cr,Sr,~,hankel] = reduceLTI(d,A,B,C,L,R);
Gamma_pr = R*R';

prec = inv(Gamma_pr);
Gpos = zeros(d,d,length(r_vals)+1);
for rr = 1:length(r_vals)
    r = r_vals(rr);
    switch method
        case 'lyap'
            LHBT = lyapchol(Ar(1:r,1:r)',Cr(:,1:r)'/sig_obs)';
            H_BT = Sr(:,1:r)*(LHBT*LHBT')*Sr(:,1:r)';
        case 'direct'
            [~,H_red] = getGH(obs_times,Cr(:,1:r),A(1:r,1:r),sig_obs);
            H_BT = Sr(:,1:r)*H_red*Sr(:,1:r)';
    end
    Gpos(:,:,rr) = inv(H_BT+ prec);
end
Gpos(:,:,end) = inv(L*L'+prec);

info.Ar = Ar;
info.Cr = Cr;
info.Sr = Sr;