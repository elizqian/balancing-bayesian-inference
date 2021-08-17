% this script compares posterior covariances for Spantini vs BT approach
% using (H, P_inf^-1) as the pencil for defining the posterior
% covariance approximation and the oblique projector for BT/Spantini means
%
% 

clear; close all

%% setup
load('heatmodel.mat')       % load LTI operators
d = size(A,1);
B = eye(d);                 % makes Pinf better conditioned than default B
% B(1:10,1:10) = diag(1000*ones(10,1));
C = zeros(5,197);           % makes for slightly slower GEV decay than default C
C(1:5,10:10:50) = eye(5);
d_out = size(C,1);


% define time for Euler
dt = 1;
T = 600;                
t = 0:dt:T;

%  define observation times and noise
n = 100;
k = (length(t)-1)/n;
obs_inds = k+1:k:length(t);
obs_times = t(k+1:k:end);
sig_obs = 0.04;

% compute compatible prior using A, B
L_pr = lyapchol(A,B)';
Gamma_pr = L_pr*L_pr';

% generate data
[G_full,H_full] = getGH(obs_times,C,A,sig_obs);
x0 = L_pr*randn(d,1);
y = G_full*x0 + sig_obs*randn(n*d_out,1);
full_rhs = G_full'*(y/sig_obs^2);
mupos_true = (H_full + inv(Gamma_pr))\full_rhs;

% compute noise-aware Fisher info
[Q,R] = qr(G_full/sig_obs,0);
L_fish = R'*Q';
% L_fish = lyapchol(A',C'/sig_obs)';
% Q_fish = L_fish*L_fish';

%% compute posterior quantities and errors
% compute Spantini posterior covariance
r_vals = 1:50;
[Gpos_sp, tau2, P_sp] = spantini_poscov(L_fish,L_pr,r_vals,'svd');

% compute BT posterior covariance
[Gpos_BT, hankel, BTinfo] = BTpos(A,B,C,sig_obs,L_fish,L_pr,r_vals,obs_times);
Gpos_BT = Gpos_sp;

% compute Forstner distances and class2 means
f_dist = zeros(length(r_vals),2);
temp = zeros(length(r_vals),2);
[mu_LRU, mu_LR, mu_BT] = deal(zeros(d,length(r_vals)));
for rr = 1:length(r_vals)
    r = r_vals(rr);
    f_dist(rr,1) = forstner(Gpos_sp(:,:,rr),Gpos_sp(:,:,end));
    f_dist(rr,2) = forstner(Gpos_BT(:,:,rr),Gpos_BT(:,:,end));
    temp(rr) = sum(log(1./(1+tau2)).^2) - sum(log(1./(1+tau2(1:r))).^2);
    
    mu_LRU(:,rr) = Gpos_sp(:,:,rr)*full_rhs;
    mu_LR(:,rr) = Gpos_sp(:,:,rr)*P_sp(:,:,rr)'*full_rhs;
    Gr = getGH(obs_times,BTinfo.Cr(:,1:r),BTinfo.Ar(1:r,1:r),sig_obs);
    mu_BT(:,rr) = Gpos_BT(:,:,rr)*BTinfo.Sr(:,1:r)*Gr'*(y/sig_obs^2);
end
mupos_ref = Gpos_sp(:,:,end)*full_rhs;

%% plots
% plot posterior covariance Forstner errors
figure(1); clf
semilogy(r_vals,f_dist(:,1),'+'); hold on
semilogy(r_vals,f_dist(:,2),'o')
semilogy(temp,'k','linewidth',1)
legend({'Spantini low-rank update','Balanced truncation','Optimal distance'},...
    'interpreter','latex','fontsize',14,'location','best')
legend boxoff
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('Error in F\"orstner metric','interpreter','latex','fontsize',14)
title('Setting 2: Posterior covariance approximation error','interpreter','latex','fontsize',16)
savePDF('figs/c2_covs',[5 4],[0 0])

% % plot posterior mean errors
err_LRU = mu_LRU - mupos_ref;
err_LR = mu_LR - mupos_ref;
err_BT = mu_BT - mupos_ref;
% figure(2); clf
% semilogy(r_vals,sqrt(sum(err_LR.^2))/norm(mupos_ref)); hold on
% semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_ref))
% semilogy(rank(H_full)*[1 1], [1e-12 1e2],'k:')
% title('Posterior covariance approximation error','interpreter','latex','fontsize',16)
% xlabel('$r$','interpreter','latex','fontsize',14)
% ylabel('$\ell^2$-error','interpreter','latex','fontsize',14)
% legend({'Spantini low-rank mean','Balanced truncation','rank of $H$'},'interpreter','latex','fontsize',14,'location','best')
% title('Posterior mean approximation error')
% legend boxoff
% savePDF('figs/c1_LRmeans',[5 4],[0 0])


figure(3); clf
semilogy(r_vals,sqrt(sum(err_LR.^2))/norm(mupos_ref)); hold on
semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_ref))
semilogy(r_vals,sqrt(sum(err_LRU.^2))/norm(mupos_ref))
semilogy(rank(H_full)*[1 1], [1e-12 1e2],'k:')
title('Posterior covariance approximation error','interpreter','latex','fontsize',16)
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('$\ell^2$-error','interpreter','latex','fontsize',14)
legend({'Spantini low-rank mean','Balanced truncation','Spantini low-rank update mean','rank of $H$'},'interpreter','latex','fontsize',14,...
    'location','best')
title('Setting 2: Posterior mean approximation error','interpreter','latex','fontsize',16)
legend boxoff
savePDF('figs/c2_allmeans',[5 4],[0 0])

figure(4); clf
semilogy(hankel,'k'); hold on; semilogy(sqrt(tau2))
legend({'HSVs','$\delta_i$ Spantini'},'interpreter','latex','fontsize',14)
title('Setting 1: Hankel singular values','interpreter','latex','fontsize',16)
legend boxoff
xlim([0 50])
savePDF('figs/c2_hankel_decay',[5 4],[0 0])
