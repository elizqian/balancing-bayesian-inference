% compares posterior covariances for:
% 1 - Spantini update using (H, P_inf^-1) pencil
% 2 - BT approx using (Q_inf, P_inf^-1) 
% 3 - BT approx using (H, P_inf^-1)
%

clear; close all

%% setup
model = 'heat'; % heat, CD, beam, build, iss1R

switch model
    case 'heat'
        load('heatmodel.mat')       % load LTI operators
        d = size(A,1);
        B = eye(d);                 % makes Pinf better conditioned than default B
        C = zeros(5,d);           % makes for slightly slower GEV decay than default C
        C(1:5,10:10:50) = eye(5);
    case 'CD'
        load('CDplayer.mat')
        d = size(A,1);
    case 'beam'
        load('beam.mat')
        d = size(A,1);
        B = eye(d);
    case 'iss1R'
        load('iss1R.mat')
        d = size(A,1);
    case 'build'
        load('build.mat')
        d = size(A,1);
end

d_out = size(C,1);

% define measurement times and noise
n       = 100;
dt_obs  = 0.1;       % making this bigger makes Spantini eigvals decay faster
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

QHerr = norm(Q_inf-dt_obs*H,'fro')/norm(Q_inf,'fro');
disp(['dt*H - Q: ',num2str(QHerr)])

%% draw random IC and generate measurements, compute true posterior
x0 = L_pr*randn(d,1);
y = G*x0 + sig_obs*randn(n*d_out,1);

full_rhs    = G'*(y/sig_obs^2);

L_prinv=inv(L_pr); 
prec_pr =L_prinv'*L_prinv;      % Is prec_pr necessary ? 
 
R_posinv=qr([G/sig_obs; L_prinv]);
R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
R_pos_true=inv(R_posinv);
Gpos_true=R_pos_true*R_pos_true';
mupos_true = R_posinv\(R_posinv'\full_rhs);

%% compute posterior approximations and errors
r_vals = 1:48;
rmax = max(r_vals);

% spantini computation as described in Remark 4
[~,S,W] = svd(G*L_pr/sig_obs);    
tau = diag(S);
What = L_pr*W;
Wtilde = L_pr'\W;

%% balancing with Q_infty
[V,S,W] = svd(L_Q'*L_pr); 
S = S(1:rmax,1:rmax);
delQ     = diag(S);
Siginvsqrt=diag(1./sqrt(delQ));
Sr      = (Siginvsqrt*V(:,1:rmax)'*L_Q')';
Tr      = L_pr*W(:,1:rmax)*Siginvsqrt;
A_BTQ    = Sr'*A*Tr;
C_BTQ    = C*Tr;

%% balancing with H
[~,R] = qr(G/sig_obs); % compute a square root factorization of H
LG = R';
[V,S,W] = svd(LG'*L_pr);
V = V(:,1:rmax);
W = W(:,1:rmax);
S = S(1:rmax,1:rmax);
delH     = diag(S);
Siginvsqrt=diag(1./sqrt(delH));
SrH      = (Siginvsqrt*V(:,1:rmax)'*LG')';
TrH      = L_pr*W(:,1:rmax)*Siginvsqrt;
A_BTH    = SrH'*A*TrH;
C_BTH    = C*TrH;

%% compute posterior approximations
f_dist = zeros(length(r_vals),4);
[mu_LRU, mu_LR, mu_BTQ, mu_BTH] = deal(zeros(d,length(r_vals)));
for rr = 1:length(r_vals)
    r = r_vals(rr);
    
% Spantini approx posterior covariance
    Rpos_sp = What*diag(sqrt([1./(1+tau(1:r).^2); ones(d-r,1)]));
    Gpos_sp=What*diag([1./(1+tau(1:r).^2); ones(d-r,1)])*What';

%%
    f_dist(rr,1) = forstner(Rpos_sp,R_pos_true,'sqrt');
    f_dist(rr,4) = sum(log(1./(1+tau(r+1:end).^2)).^2);
    
    % Spantini approx posterior means
    Pi_r = What(:,1:r)*Wtilde(:,1:r)';
    mu_LRU(:,rr) = Gpos_sp*full_rhs;
    mu_LR(:,rr)  = Gpos_sp*Pi_r'*full_rhs;
    
    % Balancing with Q_infty - generate G_BT,H_BT
    G_BTQ = zeros(n*d_out,r);
    iter = expm(A_BTQ(1:r,1:r)*dt_obs);
    temp = C_BTQ(:,1:r);
    for i = 1:n
        temp = temp*iter;
        G_BTQ((i-1)*d_out+1:i*d_out,:) = temp;
    end
    G_BTQ = G_BTQ*Sr(:,1:r)';
    H_BTQ = G_BTQ'*G_BTQ/sig_obs^2;

    % Balancing with Q_infty - compute posterior covariance and mean
    R_posinv=qr([G_BTQ/sig_obs; L_prinv]);
    R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
    R_pos_BTQ=inv(R_posinv);
    Gpos_BTQ=R_pos_BTQ*R_pos_BTQ';
    
    f_dist(rr,2) = forstner(R_pos_BTQ,R_pos_true,'sqrt');
    mu_BTQ(:,rr) = Gpos_BTQ*G_BTQ'*(y/sig_obs^2);
    
    % Balancing with H - generate G_BT, H_BT
    G_BTH = zeros(n*d_out,r);
    iter = expm(A_BTH(1:r,1:r)*dt_obs);
    temp = C_BTH(:,1:r);
    for i = 1:n
        temp = temp*iter;
        G_BTH((i-1)*d_out+1:i*d_out,:) = temp;
    end
    G_BTH = G_BTH*SrH(:,1:r)';
    H_BTH = G_BTH'*G_BTH/sig_obs^2;

    % Balancing with H - compute posterior covariance and mean
    R_posinv=qr([G_BTH/sig_obs; L_prinv]);
    R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
    R_pos_BTH=inv(R_posinv);
    Gpos_BTH=R_pos_BTH*R_pos_BTH';
       
    f_dist(rr,3) = forstner(R_pos_BTH,R_pos_true,'sqrt');
    mu_BTH(:,rr) = Gpos_BTH*G_BTH'*(y/sig_obs^2);
end

%% plots
% Warning if complex parts of Förstner distances are nontrivial
if ~isempty(find(abs(imag(f_dist))>eps*abs(real(f_dist))))
    warning('Significant imaginary parts found in Forstner distance')
end
% Otherwise imaginary parts are trivial artifacts of generalized eig
   f_dist=real(f_dist);
 
% plot posterior covariance Forstner errors
figure(1); clf
semilogy(r_vals,f_dist(:,1)); hold on
semilogy(r_vals,f_dist(:,2),'o')
semilogy(r_vals,f_dist(:,3),'x')
semilogy(r_vals,f_dist(:,4),'k:')
legend({'Spantini low-rank update','BT with Q','BT with H'},...
    'interpreter','latex','fontsize',14,'location','best')
legend boxoff
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('Error in F\"orstner metric','interpreter','latex','fontsize',14)
title(['Posterior covariance: $T = ',num2str(n*dt_obs),', \Delta t = ',num2str(dt_obs),'$'],'interpreter','latex','fontsize',16)
savePDF(['figs/',model,'_T',num2str(n*dt_obs),'_dt',num2str(dt_obs),'_cov'],[5 4],[0 0])

% plot posterior mean errors
err_LRU = mu_LRU - mupos_true;
err_LR = mu_LR - mupos_true;
err_BT = mu_BTQ - mupos_true;
err_BT2 = mu_BTH - mupos_true;

figure(2); clf
semilogy(r_vals,sqrt(sum(err_LR.^2))/norm(mupos_true)); hold on
semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_true))
semilogy(r_vals,sqrt(sum(err_BT2.^2))/norm(mupos_true))
semilogy(r_vals,sqrt(sum(err_LRU.^2))/norm(mupos_true))
ylim([1e-5 1e1])
xlabel('$r$','interpreter','latex','fontsize',14)
ylabel('Relative $\ell^2$-error','interpreter','latex','fontsize',14)
legend({'Spantini low-rank mean','BT with Q','BT with H','Spantini low-rank update mean'},'interpreter','latex','fontsize',14,...
    'location','best')
title(['Posterior means: $T = ',num2str(n*dt_obs),', \Delta t = ',num2str(dt_obs),'$'],'interpreter','latex','fontsize',16)
legend boxoff
savePDF(['figs/',model,'_T',num2str(n*dt_obs),'_dt',num2str(dt_obs),'_means'],[5 4],[0 0])

figure(3); clf
semilogy(tau,'+'); hold on
semilogy(delQ,'o')
semilogy(delH,'x')
xlabel(['$\|\Delta t\cdot H - Q_\infty\|_F / \|Q_\infty\|_F = ',num2str(QHerr),'$'],'fontsize',14,'interpreter','latex')
legend({'Spantini: $(H,\Gamma_{pr}^{-1})$','Balancing: $(Q_\infty,\Gamma_{pr}^{-1})$','Balancing: $(H,\Gamma_{pr}^{-1})$'},'interpreter','latex','fontsize',14)
legend boxoff
title('Hankel singular values/sqrt of Spantini GEVs','interpreter','latex','fontsize',16)
xlim([0 rmax])
savePDF(['figs/',model,'_T',num2str(n*dt_obs),'_dt',num2str(dt_obs),'_gev'],[5 4],[0 0])
