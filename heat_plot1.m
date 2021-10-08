clear; close all

heat_setup1

%% compute true posterior
full_rhs    = G'*(y./(sig_obs_long.^2));

L_prinv=inv(L_pr); 
 
R_posinv=qr([Go; L_prinv],0);
R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
R_pos_true=inv(R_posinv);
Gpos_true=R_pos_true*R_pos_true';
mupos_true = R_posinv\(R_posinv'\full_rhs);

%% compute posterior approximations and errors
r_vals = 1:20;
rmax = max(r_vals);

% (H,Gamma_pr^-1) computations
[~,R] = qr(Go,0); % compute a square root factorization of H
LG = R';
[V,S,W] = svd(LG'*L_pr,0);
tau = diag(S);
What = L_pr*W;      % spantini directions
Wtilde = L_pr'\W;
S = S(1:rmax,1:rmax);   
delH     = diag(S);
Siginvsqrt=diag(1./sqrt(delH));
SrH      = (Siginvsqrt*V(:,1:rmax)'*LG')';
TrH      = L_pr*W(:,1:rmax)*Siginvsqrt;
A_BTH    = SrH'*A*TrH;
C_BTH    = C*TrH;

%% balancing with Q_infty
[V,S,W] = svd(L_Q'*L_pr); 
S = S(1:rmax,1:rmax);
delQ     = diag(S);
Siginvsqrt=diag(1./sqrt(delQ));
Sr      = (Siginvsqrt*V(:,1:rmax)'*L_Q')';
Tr      = L_pr*W(:,1:rmax)*Siginvsqrt;
A_BTQ    = Sr'*A*Tr;
C_BTQ    = C*Tr;

%% compute posterior approximations
f_dist = zeros(length(r_vals),4);
[mu_LRU, mu_LR, mu_BTQ, mu_BTH] = deal(zeros(d,length(r_vals)));
for rr = 1:length(r_vals)
    r = r_vals(rr);
    
    % Spantini approx posterior covariance
    Rpos_sp = What*diag(sqrt([1./(1+tau(1:r).^2); ones(d-r,1)]));
    Gpos_sp=What*diag([1./(1+tau(1:r).^2); ones(d-r,1)])*What';

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
    G_BTQo = G_BTQ./sig_obs_long;
    H_BTQ = G_BTQo'*G_BTQo;

    % Balancing with Q_infty - compute posterior covariance and mean
    R_posinv=qr([G_BTQo; L_prinv],0);
    R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
    R_pos_BTQ=inv(R_posinv);
    Gpos_BTQ=R_pos_BTQ*R_pos_BTQ';
    
    f_dist(rr,2) = forstner(R_pos_BTQ,R_pos_true,'sqrt');
    mu_BTQ(:,rr) = Gpos_BTQ*G_BTQ'*(y./(sig_obs_long.^2));
    
    % Balancing with H - generate G_BT, H_BT
    G_BTH = zeros(n*d_out,r);
    iter = expm(A_BTH(1:r,1:r)*dt_obs);
    temp = C_BTH(:,1:r);
    for i = 1:n
        temp = temp*iter;
        G_BTH((i-1)*d_out+1:i*d_out,:) = temp;
    end
    G_BTH = G_BTH*SrH(:,1:r)';
    G_BTHo = G_BTH./sig_obs_long;
    H_BTH = G_BTHo'*G_BTHo;

    % Balancing with H - compute posterior covariance and mean
    R_posinv=qr([G_BTHo; L_prinv],0);
    R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
    R_pos_BTH=inv(R_posinv);
    Gpos_BTH=R_pos_BTH*R_pos_BTH';
       
    f_dist(rr,3) = forstner(R_pos_BTH,R_pos_true,'sqrt');
    mu_BTH(:,rr) = Gpos_BTH*G_BTH'*(y./(sig_obs_long.^2));
end

%% plots
% Warning if complex parts of Förstner distances are nontrivial
if ~isempty(find(abs(imag(f_dist))>eps*abs(real(f_dist))))
    warning('Significant imaginary parts found in Forstner distance')
end
% Otherwise imaginary parts are trivial artifacts of generalized eig
   f_dist=real(f_dist);
 
%% plot posterior covariance Forstner errors
figure(11); clf
semilogy(r_vals,f_dist(:,1)); hold on
semilogy(r_vals,f_dist(:,2),'o')
semilogy(r_vals,f_dist(:,3),'x')
legend({'Spantini','BT-Q','BT-H'},...
    'interpreter','latex','fontsize',18,'location','best')
legend boxoff
grid on
xlabel('$r$','interpreter','latex','fontsize',18)
title(['F\"orstner posterior covariance error'],'interpreter','latex','fontsize',20)
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_cov'],[4.5 4],[0 0])

%% plot posterior mean errors
err_LRU = mu_LRU - mupos_true;
err_LR = mu_LR - mupos_true;
err_BT = mu_BTQ - mupos_true;
err_BT2 = mu_BTH - mupos_true;

figure(12); clf
semilogy(r_vals,sqrt(sum(err_LR.^2))/norm(mupos_true)); hold on
semilogy(r_vals,sqrt(sum(err_LRU.^2))/norm(mupos_true),'Color',[0.4940    0.1840    0.5560])
semilogy(r_vals,sqrt(sum(err_BT.^2))/norm(mupos_true),'o','Color',getColor(2))
semilogy(r_vals,sqrt(sum(err_BT2.^2))/norm(mupos_true),'x','Color',getColor(3))
ylim([1e-10 1e0])
xlabel('$r$','interpreter','latex','fontsize',18)
legend({'Spantini LR','Spantini LRU','BT-Q','BT-H'},'interpreter','latex','fontsize',18,...
    'location','southwest')
title(['Relative $\ell^2$ posterior mean error'],'interpreter','latex','fontsize',20)
legend boxoff
grid on
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_mean1'],[4.5 4],[0 0])

%% plot Gpos^-1 norm
poserr_LR = R_pos_true\err_LR;
poserr_LRU = R_pos_true\err_LRU;
poserr_BT = R_pos_true\err_BT;
poserr_BT2 = R_pos_true\err_BT2;

posnorm = zeros(rmax,4);
posnorm(:,1) = sqrt(sum(poserr_LR.^2));
posnorm(:,2) = sqrt(sum(poserr_LRU.^2));
posnorm(:,3) = sqrt(sum(poserr_BT.^2));
posnorm(:,4) = sqrt(sum(poserr_BT2.^2));
posnormref = norm(R_pos_true\mupos_true);

figure(121); clf
semilogy(r_vals,posnorm(:,1)/posnormref); hold on
semilogy(r_vals,posnorm(:,2)/posnormref,'Color',[0.4940    0.1840    0.5560]); 
semilogy(r_vals,posnorm(:,3)/posnormref,'o','Color',getColor(2))
semilogy(r_vals,posnorm(:,4)/posnormref,'x','Color',getColor(3))
xlabel('$r$','interpreter','latex','fontsize',18)
legend({'Spantini LR','Spantini LRU','BT-Q','BT-H'},'interpreter','latex','fontsize',18,...
    'location','northeast')
title(['Relative $\Gamma_{\rm pos}^{-1}$-norm posterior mean error'],'interpreter','latex','fontsize',20)
legend boxoff
ylim([1e-10 1e0])
grid on
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_posnormmean1'],[4.5 4],[0 0])
%% plot HSVs
figure(13); clf
semilogy(r_vals,delQ/delQ(1),'o','Color',[0.8500    0.3250    0.0980]); hold on
semilogy(r_vals,delH/delH(1),'x','Color',[0.9290    0.6940    0.1250]); hold on
grid on
title(['(Approximate) Hankel singular values'],...
   'interpreter','latex','fontsize',20)
legend({'$\delta_i$','$\tau_i$'},'interpreter','latex','fontsize',18,'location','southwest')
legend boxoff
xlabel('$i$','interpreter','latex')
ylim([1e-10 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_HSV1'],[4.5 4],[0 0])