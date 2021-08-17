% compares posterior covariances for:
% 1 - Spantini update using (H, P_inf^-1) pencil
% 2 - BT approx using (H, P_inf^-1) 
%
% BT models for r close to r_max end up unstable; Serkan says he can fix
% it?

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
dt_obs  = 1;       % making this bigger makes Spantini eigvals decay faster
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

% compute Forstner distances and class2 means
f_dist = zeros(length(r_vals),2);
temp = zeros(length(r_vals),2);
[mu_LRU, mu_LR, mu_BT] = deal(zeros(d,length(r_vals)));
stable = ones(length(r_vals),1);
for rr = 1:length(r_vals)
    r = r_vals(rr);
    
    % Spantini approx posterior covariance
    KKT = What(:,1:r)*diag(tau(1:r).^2./(1+tau(1:r).^2))*What(:,1:r)';
    Gpos_sp = Gamma_pr - KKT;
    f_dist(rr,1) = forstner(Gpos_sp,Gpos_true);
    
    % Spantini approx posterior means
    Pi_r = What(:,1:r)*Wtilde(:,1:r)';
    mu_LRU(:,rr) = Gpos_sp*full_rhs;
    mu_LR(:,rr)  = Gpos_sp*Pi_r'*full_rhs;
    
    % generate G_BT, H_BT
    G_BT = zeros(n*d_out,r);
    iter = expm(A_BT(1:r,1:r)*dt_obs);
    if any(real(eig(A_BT(1:r,1:r)))>0)
        stable(rr) = 0;
    end
    temp = C_BT(:,1:r);
    for i = 1:n
        temp = temp*iter;
        G_BT((i-1)*d_out+1:i*d_out,:) = temp;
    end
    G_BT = G_BT*Sr(:,1:r)';
    H_BT = G_BT'*G_BT/sig_obs^2;

    % BT approx posterior covariance
    Gpos_BT = inv(H_BT+ prec_pr);
    if ~any(isnan(H_BT),'all') && ~any(isinf(H_BT),'all')
        f_dist(rr,2) = forstner(Gpos_BT,Gpos_true);
    end
    
    % BT approx posterior mean
    mu_BT(:,rr) = Gpos_BT*G_BT'*(y/sig_obs^2);
end
% show flags for whether BT model is stable
stable' 
%% plots
% plot posterior covariance Forstner errors
figure(1); clf
semilogy(r_vals,f_dist(:,1)); hold on
semilogy(r_vals,f_dist(:,2),'x','Color',[0.9290    0.6940    0.1250])
legend({'Spantini low-rank update','BT with H'},...
    'interpreter','latex','fontsize',14)
legend boxoff
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('Error in F\"orstner metric','interpreter','latex','fontsize',14)
title('Posterior covariance approximation error','interpreter','latex','fontsize',16)

% plot posterior mean errors
err_LRU = mu_LRU - mupos_true;
err_LR = mu_LR - mupos_true;
err_BT = mu_BT - mupos_true;

figure(2); clf
semilogy(r_vals,sqrt(sum(err_LR.^2))/norm(mupos_true)); hold on
semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_true),'Color',[0.9290    0.6940    0.1250])
semilogy(r_vals,sqrt(sum(err_LRU.^2))/norm(mupos_true),'Color',[0.4940    0.1840    0.5560])
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('$\ell^2$-error','interpreter','latex','fontsize',14)
legend({'Spantini low-rank mean','BT with H','Spantini low-rank update mean'},'interpreter','latex','fontsize',14,...
    'location','best')
title('Posterior mean approximation error','interpreter','latex','fontsize',16)
legend boxoff

figure(3); clf
semilogy(tau,'+'); hold on
semilogy(del,'x','Color',[0.9290    0.6940    0.1250])
legend({'Spantini: $(H,\Gamma_{pr}^{-1})$','Balancing: $(H,\Gamma_{pr}^{-1})$'},'interpreter','latex','fontsize',14)
legend boxoff
title('Hankel singular values/sqrt Spantini GEVs','interpreter','latex','fontsize',16)
xlim([0 rmax])

function nm = forstner(A,B)
    sig = eig(A,B);
    nm = sum(log(sig).^2);
end
