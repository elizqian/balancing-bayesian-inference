% compares scaled H to Q_fish

clear; close all

%% setup
% model: heat, CD, beam, build, iss1R
model = 'heat';

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

eigA = eig(full(A));
temp = 1/min(abs(real(eigA)));
ct = round(temp)    % characteristic time

% noise level
sig_obs = 0.04;

% compute Gramians
L_pr = lyapchol(A,B)';  
Gamma_pr = L_pr*L_pr';
L_Q = lyapchol(A',C'/sig_obs)';
Q_inf = L_Q*L_Q';

rmax = 40;

% final time
T_vals = [0.5 1 2]*ct;
dt_vals = [1 .1 1e-2];
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
        [~,S,W] = svd(G*L_pr/sig_obs,'econ');
        r = min(rmax,size(S,1)); 
        delH     = diag(S(1:r,1:r));
        Siginvsqrt=diag(1./sqrt(delH));
        TrH      = L_pr*W(:,1:r)*Siginvsqrt;
        HSVH(tt,nn,1:r) = delH;
        
        % compute GEV - Q
        [~,S,W] = svd(L_Q'*L_pr); 
        delQ     = diag(S(1:r,1:r));
        Siginvsqrt=diag(1./sqrt(delQ));
        TrQ      = L_pr*W(:,1:r)*Siginvsqrt;
        HSVQ(tt,nn,1:r) = delQ;
        
        % compute Q vs. H subspace angles
        [Uh,~,~]=svd(H);
        [Uq,~,~]=svd(Q_inf);

        for k=1:r
            angle1(tt,nn,k) = subspace(Uh(:,1:k),Uq(:,1:k));
            angle2(tt,nn,k) = subspace(TrH(:,1:k),TrQ(:,1:k));
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
savePDF(['figs/',model,'_QHfronorm'],[6 5],[0 0])

% subspace angle between Q and H
figure(2); clf
for i = 1:length(T_vals)
    for j = 1:length(dt_vals)
        subplot(length(T_vals),length(dt_vals),(j-1)*length(T_vals)+i)
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
savePDF(['figs/',model,'_angles'],[6 5],[0 0])

% HSVs
ymin = min([min(HSVH,[],'all'), min(HSVQ,[],'all')]);
ymax = max([max(HSVH,[],'all'), max(HSVQ,[],'all')]);
figure(3); clf
for i = 1:length(T_vals)
    for j = 1:length(dt_vals)
        subplot(length(T_vals),length(dt_vals),(j-1)*length(T_vals)+i)
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
savePDF(['figs/',model,'_hsvs'],[6 5],[0 0])
