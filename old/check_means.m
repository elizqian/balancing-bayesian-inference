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
n = 10;
k = (length(t)-1)/n;
obs_inds = k+1:k:length(t);
obs_times = t(k+1:k:end);
sig_obs = 0.04;

% compute compatible prior using A, B
L_pr = lyapchol(A,B)';
Gamma_pr = L_pr*L_pr';

% generate data
tic
[G_full,H_full] = getGH(obs_times,C,A,sig_obs,dt);
tocfull = toc;
x0 = L_pr*randn(d,1);
y = G_full*x0 + sig_obs*randn(n*d_out,1);
full_rhs = G_full'*(y/sig_obs^2);
mupos_true = (H_full + inv(Gamma_pr))\full_rhs;

% compute noise-aware Fisher info
L_fish = lyapchol(A',C'/sig_obs)';
Q_fish = L_fish*L_fish';

%% compute posterior quantities and errors
% compute Spantini posterior covariance
r_vals = 1:50;
[Gpos_sp, tau2, P_sp] = spantini_poscov(L_fish,L_pr,r_vals,'svd');
[Gpos_sp2, tau3, P_sp2] = spantini_poscov(Q_fish,Gamma_pr,r_vals,'eig');
[mu44, del2] = lowrankmean(y,G_full,sig_obs,L_pr,r_vals); % 4.4 from Spantini

 %4.6 from Spantini
 Tikh = (L_pr'*(G_full'*G_full)*L_pr/sig_obs^2+eye(197))\L_pr'*G_full'/sig_obs;
 [myV,myS,myW] = svd(Tikh);
 
 mu46 = zeros(d,length(r_vals));
 mu47 = zeros(d,length(r_vals));

% compute BT posterior covariance
[Gpos_BT, hankel, BTinfo] = BTpos(A,B,C,sig_obs,L_fish,L_pr,r_vals);

% compute class2 means


[mu229, mu_BT] = deal(zeros(d,length(r_vals)));
for rr = 1:length(r_vals)
    r = r_vals(rr);
    
    A46(:,:,r) = L_pr*myV(:,1:r)* myS(1:r,1:r) * myW(:,1:r)'/sig_obs;
    mu46(:,r) = A46(:,:,r)*y;
    mu47(:,r) = Gpos_sp(:,:,r)*full_rhs;
    mu229(:,r) = Gpos_sp(:,:,r)*P_sp(:,:,r)*full_rhs;

    A228(:,:,r) = (inv(Gamma_pr)+P_sp(:,:,r)*(G_full'*G_full)*P_sp(:,:,r)'/sig_obs^2)\P_sp(:,:,r)*G_full'/sig_obs^2;
    A220(:,:,r) = (inv(Gamma_pr)+P_sp(:,:,r)*(G_full'*G_full)*P_sp(:,:,r)'/sig_obs^2)\(P_sp(:,:,r));
    A230(:,:,r) = (inv(Gamma_pr)+P_sp(:,:,r)*H_full*P_sp(:,:,r)')\(P_sp(:,:,r));
    mu220(:,r) = A220(:,:,r)*full_rhs;
    mu230(:,r) = A230(:,:,r)*full_rhs;
    mu228(:,r) = A228(:,:,r)*y;
    Gr = getGH(obs_times,BTinfo.Cr(:,1:r),BTinfo.Ar(1:r,1:r),sig_obs,dt);
    mu_BT(:,r) = Gpos_BT(:,:,r)*BTinfo.Sr(:,1:r)*Gr'*(y/sig_obs^2);
end
mupos_ref = Gpos_sp(:,:,end)*full_rhs;


%% plot posterior mean errors
err_229= mu229 - mupos_ref;
err_47= mu47 - mupos_ref;
err_44 = mu44 - mupos_true;
err_46 = mu46 - mupos_true;
err_BT = mu_BT - mupos_ref;
err_228 = mu228 - mupos_true;

figure(2); clf
semilogy(r_vals,sqrt(sum(err_229.^2))/norm(mupos_ref)); hold on
semilogy(r_vals,sqrt(sum(err_228.^2))/norm(mupos_true)); 
semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_ref))
semilogy(r_vals,sqrt(sum(err_47.^2))/norm(mupos_ref))
title('Posterior covariance approximation error','interpreter','latex','fontsize',16)
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('$\ell^2$-error','interpreter','latex','fontsize',14)
legend({'2.28 Gpos','2.28 compute','Balanced truncation','4.7 Sp'},'interpreter','latex','fontsize',14)
title('Posterior mean approximation error')
legend boxoff
%savePDF('figs/c1_LRUmeans',[5 4],[0 0])


figure(3); clf
semilogy(r_vals,sqrt(sum(err_229.^2))/norm(mupos_ref)); hold on
semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_ref))
semilogy(r_vals,sqrt(sum(err_47.^2))/norm(mupos_ref))
semilogy(r_vals,sqrt(sum(err_44.^2))/norm(mupos_true))
semilogy(r_vals,sqrt(sum(err_46.^2))/norm(mupos_true))
semilogy(r_vals,sqrt(sum(err_228.^2))/norm(mupos_true)); 
title('Posterior covariance approximation error','interpreter','latex','fontsize',16)
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('$\ell^2$-error','interpreter','latex','fontsize',14)
legend({'2.28','Balanced truncation','4.7 Sp','4.4 Sp','4.6 Sp'},'interpreter','latex','fontsize',14)
title('Posterior mean approximation error')

%%
figure(1);clf
 plot(mu46(:,10))
hold on
plot(mu47(:,10))
plot(mu229(:,10))
plot(mu228(:,10))
plot(mu220(:,10))
plot(mu_BT(:,10))
plot(mu230(:,10))
legend('46','47','229','228','220','BT','230')