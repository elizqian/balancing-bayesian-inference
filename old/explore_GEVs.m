% looks at GEV dependence on different obs models

clear; close all

%% setup
load('heatmodel.mat')       % load LTI operators
d = size(A,1);
B = eye(d);                 % makes Pinf better conditioned than default B
C = zeros(5,197);           % makes for slightly slower GEV decay than default C
C(1:5,10:10:50) = eye(5);
d_out = size(C,1);

% define measurement times and noise
n       = 100;
dt_obs  = 10;       % making this bigger makes Spantini eigvals decay faster
obs_times = dt_obs:dt_obs:n*dt_obs;
sig_obs = 0.04;

% compute Gramians
L_pr = lyapchol(A,B)';
Gamma_pr = L_pr*L_pr';
L_Q = lyapchol(A',C'/sig_obs)';
Q_inf = L_Q*L_Q';

% define full forward model and Fisher info
G = zeros(n*d_out,d);
iter = expm(A*dt_obs);
temp = C;
for i = 1:n
    temp = temp*iter;
    G((i-1)*d_out+1:i*d_out,:) = temp;
end
H = G'*G/sig_obs^2;

%% draw random IC and generate measurements, compute true posterior
x0 = L_pr*randn(d,1);
y = G*x0 + sig_obs*randn(n*d_out,1);

full_rhs    = G'*(y/sig_obs^2);
prec_pr     = inv(Gamma_pr);
Gpos_true   = inv(H + prec_pr);
mupos_true  = Gpos_true*full_rhs;

%% compute posterior approximations and errors
r_vals = 1:50;
rmax = max(r_vals);

% spantini computation as described in Remark 4
[~,S,W] = svd(G*L_pr/sig_obs);    
tau = diag(S);
What = L_pr*W;
Wtilde = Gamma_pr\What; 

% balancing transformation and balanced ops
R = qr(G/sig_obs); % compute a square root factorization of H
LG = R';
[V,S,W] = svd(LG(:,1:rmax)'*L_pr(:,1:rmax));     
del     = diag(S);
Siginvsqrt = S^-0.5;
Sr      = (Siginvsqrt*V'*LG(:,1:rmax)')';
Tr      = L_pr(:,1:rmax)*W*Siginvsqrt;
A_BT    = Sr'*A*Tr;
C_BT    = C*Tr;

[V,S,W] = svd(L_Q(:,1:rmax)'*L_pr(:,1:rmax));       % differs from above in use of Chol factor of infinite Gramian
boo     = diag(S);
%% plots
figure(3); clf
semilogy(tau,'+'); hold on
semilogy(del,'o')
semilogy(boo,'x')
legend({'Spantini: $(H,\Gamma_{pr}^{-1})$','Balancing: $(H,\Gamma_{pr}^{-1})$','Balancing: $(Q_\infty,\Gamma_{pr}^{-1})$'},'interpreter','latex','fontsize',14)
legend boxoff
title(['Generalized eigenvalues: $\Delta t = ',num2str(dt_obs),'$'],'interpreter','latex','fontsize',16)
xlim([0 rmax])
savePDF(['figs/m2_eigs_dt',num2str(dt_obs)],[5 4],[0 0])

function nm = forstner(A,B)
    sig = eig(A,B);
    nm = sum(log(sig).^2);
end
