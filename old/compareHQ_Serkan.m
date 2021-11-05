% compares scaled H to Q_fish

clear; close all

setup

rmax = 30;

% final time
T_vals = [1 10 50];
dt_vals = [1e-2 1e-3 1e-4];
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
        H = G'*G/sig_obs^2;
        forst(tt,nn) = forstner(sqrt(dt_obs)*G'/sig_obs,L_Q,'sqrt');
        fro(tt,nn) = norm(dt_obs*H-Q_inf,'fro');
        
        % compute GEV - H
        %[V,S,W] = svd(G*L_pr/sig_obs,'econ');
        [~,R] = qr(G/sig_obs,0); % compute a square root factorization of H
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
        HSVQ(tt,nn,1:r) = delQ/delQ(1);;
        
        
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
figure(1); clf
str = cell(length(T_vals),1);
sty = {'-','-','-'};
for i = 1:length(T_vals)
    loglog(dt_vals,fro(i,:)/norm(Q_inf,'fro'),sty{i}); hold on
    str{i} = ['T = ',num2str(T_vals(i))];
end
xlabel('$\Delta t$','interpreter','latex','fontsize',16)
title('$||\Delta tH - Q_\infty||_F \,/\, ||Q_\infty||_F$','interpreter','latex','fontsize',18)
legend(str,'interpreter','latex','fontsize',16,'location','best'); legend boxoff
%savePDF(['figs/',model,'_QHfronorm'],[6 5],[0 0])

% subspace angle between Q and H
figure(2); clf
for i = 1:length(T_vals)
    for j = 1:length(dt_vals)
        subplot(length(dt_vals),length(T_vals),(j-1)*length(T_vals)+i)
        plot(squeeze(angle1(i,j,:)))
        hold on
        plot(squeeze(angle2(i,j,:)),':')
        ylim([0 pi/2])
        if i == 1
            ylabel(['$\Delta t = ',num2str(dt_vals(j)),'$'],'interpreter','latex','fontsize',16)
        end
        if j == length(dt_vals)
            xlabel(['$T = ',num2str(T_vals(i)),'$'],'interpreter','latex','fontsize',16)
        end
    end
end
legend('H v Q','BT H v BT Q','location','best')
sgtitle('$Q$ vs $H$ subspace angle','interpreter','latex','fontsize',18)
%savePDF(['figs/',model,'_angles'],[6 5],[0 0])
%%
% HSVs
ymin = min([min(HSVH,[],'all'), min(HSVQ,[],'all')]);
ymax = max([max(HSVH,[],'all'), max(HSVQ,[],'all')]);
figure(3); clf
for i = 1:length(T_vals)
    for j = 1:length(dt_vals)
        subplot(length(dt_vals),length(T_vals),(j-1)*length(T_vals)+i)
        semilogy(squeeze(HSVH(i,j,:)),'x'); hold on;
        semilogy(squeeze(HSVQ(i,j,:)),'o')
        ylim([ymin ymax])
        if i == 1
            ylabel(['$\Delta t = ',num2str(dt_vals(j)),'$'],'interpreter','latex','fontsize',16)
            set(gca,'ytick',logspace(-8,8,9));
        else
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
        end
        if j == length(dt_vals)
            xlabel(['$T = ',num2str(T_vals(i)),'$'],'interpreter','latex','fontsize',16)
        else
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
        end
    end
end
sgtitle('$Q$ vs $H$ HSVs','interpreter','latex','fontsize',18)
legend({'$H$','$Q$'},'interpreter','latex')
%savePDF(['figs/',model,'_hsvs'],[6 5],[0 0])
