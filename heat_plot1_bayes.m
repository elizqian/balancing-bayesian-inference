clear; close all

heat_setup1

% generate random data from multiple initial conditions and measurements
num_reps = 100;
x0_all = L_pr*randn(d,num_reps);
y_all  = G*x0_all;
m_all  = y_all + sig_obs_long.*randn(n*d_out,num_reps);

%% compute true posterior
full_rhs    = G'*(y./(sig_obs_long.^2));
full_rhs_all = G'*(y_all./(sig_obs_long.^2));

L_prinv=inv(L_pr); 
 
R_posinv=qr([Go; L_prinv],0);
R_posinv=triu(R_posinv(1:d,:)); % Pull out upper triangular factor
R_pos_true=inv(R_posinv);
Gpos_true=R_pos_true*R_pos_true';
mupos_true = R_posinv\(R_posinv'\full_rhs);
mupos_true_all = R_posinv\(R_posinv'\full_rhs_all);

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
mu_errs = zeros(length(r_vals),4);
for rr = 1:length(r_vals)
    r = r_vals(rr);
    
    % Spantini approx posterior covariance
    Rpos_sp = What*diag(sqrt([1./(1+tau(1:r).^2); ones(d-r,1)]));
    Gpos_sp=What*diag([1./(1+tau(1:r).^2); ones(d-r,1)])*What';

    f_dist(rr,1) = forstner(Rpos_sp,R_pos_true,'sqrt');
    f_dist(rr,4) = sum(log(1./(1+tau(r+1:end).^2)).^2);
    
    % Spantini approx posterior means
    Pi_r = What(:,1:r)*Wtilde(:,1:r)';
    temp = Gpos_sp*Pi_r'*full_rhs_all;
    temp = R_pos_true\(temp - mupos_true_all);
    mu_errs(rr,1) = mean(sqrt(sum(temp.^2)));
    
    temp = Gpos_sp*full_rhs_all;
    temp = R_pos_true\(temp-mupos_true_all);
    mu_errs(rr,2) = mean(sqrt(sum(temp.^2)));
    
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
    temp = Gpos_BTQ*G_BTQ'*(y_all./(sig_obs_long.^2));
    temp = R_pos_true\(temp - mupos_true_all);
    mu_errs(rr,3) = mean(sqrt(sum(temp.^2)));

    
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
    temp = Gpos_BTH*G_BTH'*(y_all./(sig_obs_long.^2));
    temp = R_pos_true\(temp - mupos_true_all);
    mu_errs(rr,4) = mean(sqrt(sum(temp.^2)));
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
legend({'OLRU','BT-Q','BT-H'},...
    'interpreter','latex','fontsize',18,'location','best')
legend boxoff
grid on
xlabel('$r$','interpreter','latex','fontsize',18)
title(['F\"orstner posterior covariance error'],'interpreter','latex','fontsize',20)
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_cov1'],[4.5 4],[0 0])

%% plot posterior mean errors in Gpos^-1 norm
posnormref = mean(sqrt(sum((R_pos_true\mupos_true_all).^2)));
figure(121); clf
semilogy(r_vals,mu_errs(:,1)/posnormref); hold on
semilogy(r_vals,mu_errs(:,2)/posnormref,'Color',[0.4940    0.1840    0.5560]); 
semilogy(r_vals,mu_errs(:,3)/posnormref,'o','Color',getColor(2))
semilogy(r_vals,mu_errs(:,4)/posnormref,'x','Color',getColor(3))
xlabel('$r$','interpreter','latex','fontsize',18)
legend({'OLR','OLRU','BT-Q','BT-H'},'interpreter','latex','fontsize',18,...
    'location','northeast')
title(['Posterior mean: normalized Bayes risk'],'interpreter','latex','fontsize',20)
legend boxoff
grid on
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_bayesmean1'],[4.5 4],[0 0])
%% plot HSVs
figure(13); clf
semilogy(r_vals,delQ/delQ(1),'o','Color',[0.8500    0.3250    0.0980]); hold on
semilogy(r_vals,delH/delH(1),'x','Color',[0.9290    0.6940    0.1250]); hold on
grid on
title(['Hankel singular values'],...
   'interpreter','latex','fontsize',20)
legend({'$\delta_i$','$\tau_i$'},'interpreter','latex','fontsize',18,'location','southwest')
legend boxoff
xlabel('$i$','interpreter','latex')
ylim([1e-10 1])
set(gca,'fontsize',16,'ticklabelinterpreter','latex')
savePDF(['paper/',model,'_HSV1'],[4.5 4],[0 0])