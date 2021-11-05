% compares scaled H to Q_fish

clear; close all

setup

rmax = 20;

% final time
T_vals = 50;
dt_vals = [.1 1e-2 1e-3 1e-4];
[forst,fro] = deal(zeros(length(T_vals),length(dt_vals)));
[angle1,angle2,HSVH,HSVQ] = deal(zeros(length(T_vals),length(dt_vals),rmax));
for tt = 1:length(T_vals)
    T = T_vals(tt);
    for nn = 1:length(dt_vals)
        dt_obs = dt_vals(nn);
        obs_times = dt_obs:dt_obs:T;
        n = length(obs_times);
        
        % define full forward model and Fisher info
        G = zeros(n*d_out,d);
        iter = expm(A*dt_obs);
        temp = C;
        for i = 1:n
            temp = temp*iter;
            G((i-1)*d_out+1:i*d_out,:) = temp;
        end
        Go = G./repmat(sig_obs,n,1);
        H = Go'*Go;
        forst(tt,nn) = forstner(sqrt(dt_obs)*Go',L_Q,'sqrt');
        fro(tt,nn) = norm(dt_obs*H-Q_inf,'fro');
        
        % compute GEV - H
        %[V,S,W] = svd(G*L_pr/sig_obs,'econ');
        [~,R] = qr(Go,0); % compute a square root factorization of H
        LG = R';
        [V,S,W] = svd(LG'*L_pr,0);
        r = min(rmax,size(S,1));
        delH     = diag(S(1:r,1:r));
        Siginvsqrt=diag(1./sqrt(delH));
%         SrH      = (Siginvsqrt*V(:,1:rmax)'*G/sig_obs)';  Use this version of SrH if Line 72 is used
        SrH      = (Siginvsqrt*V(:,1:rmax)'*LG')';
        TrH      = L_pr*W(:,1:r)*Siginvsqrt;
        HSVH(tt,nn,1:r) = delH/delH(1);
        
        % compute GEV - Q
        [V,S,W] = svd(L_Q'*L_pr);
        delQ     = diag(S(1:r,1:r));
        Siginvsqrt=diag(1./sqrt(delQ));
        SrQ      = (Siginvsqrt*V(:,1:rmax)'*L_Q')';
        TrQ      = L_pr*W(:,1:r)*Siginvsqrt;
        HSVQ(tt,nn,1:r) = delQ/delQ(1);
        
        % compute Q vs. H subspace angles
        [Uh,~,~]=svd(H);
        [Uq,~,~]=svd(Q_inf);
        
        % Try (TrH vs TrQ,  (SrH vs SrQ), or  (SrH*TrH' vs SrQ*TrQ')
        [BTH,~,~] = svd(TrH);
        [BTQ,~,~] = svd(TrQ);
        
        for k=1:r
            angle1(tt,nn,k) = subspace(Uh(:,1:k),Uq(:,1:k));
            angle2(tt,nn,k) = subspace(BTH(:,1:k),BTQ(:,1:k));
        end
        
       
    end
end
%% plots
% Frobenius norm relative difference
figure(21); clf
subplot(1,2,1)
loglog(dt_vals,fro/norm(Q_inf,'fro')); hold on
xlabel('Observation interval $\Delta t$','interpreter','latex','fontsize',16)
ylabel('$||\Delta tH - Q_\infty||_F \,/\, ||Q_\infty||_F$','interpreter','latex','fontsize',16)
title('Frobenius norm','interpreter','latex','fontsize',18)
grid on

subplot(1,2,2)
semilogx(dt_vals,squeeze(angle1(1,:,end))); hold on
semilogx(dt_vals,squeeze(angle2(1,:,end)),':')
xlabel('Observation interval $\Delta t$','interpreter','latex','fontsize',16)
ylabel('Angle (rad)','interpreter','latex','fontsize',16)
legend({'$Q$ vs. $H$','$(Q,\Gamma_{\rm pr}^{-1})$ vs. $(H,\Gamma_{\rm pr}^{-1})$'},'interpreter','latex','fontsize',14,'location','southeast')
legend boxoff
title('Dominant subspace','interpreter','latex','fontsize',18)
grid on

savePDF(['figs/',model,'_QHconv'],[10 4],[0 0])