% compares scaled H to Q_fish

clear; close all

%% setup
load('heatmodel.mat')       % load LTI operators
d = size(A,1);
B = eye(d);                 % makes Pinf better conditioned than default B
C = zeros(5,197);           % makes for slightly slower GEV decay than default C
C(1:5,10:10:50) = eye(5);
d_out = size(C,1);

% compute noise-aware infinite Gramian
sig_obs = 0.04;
L_fish = lyapchol(A',C'/sig_obs)';
Q_fish = L_fish*L_fish';

% final time
T_vals = [10 50 100 500 1000];
dt_vals = [1 0.1 0.01 0.001];
[forst,fro,compatH] = deal(zeros(length(T_vals),length(dt_vals)));
for tt = 1:length(T_vals)
    T = T_vals(tt);
    for nn = 1:length(dt_vals)
        dt_obs = dt_vals(nn);
        obs_times = dt_obs:dt_obs:T;
        
        % get Fisher info for specified observation model
        [G_full,H_full] = getGH(obs_times,C,A,sig_obs);
        forst(tt,nn) = forstner(dt_obs*H_full,Q_fish);
        fro(tt,nn) = norm(dt_obs*H_full-Q_fish,'fro');
        
        lhs = H_full*A + A'*H_full;
        try(chol(-lhs))
            disp([num2str(dt_obs), ' compatible'])
            compatH(tt,nn) = 1;
        catch ME
            temp = eig(lhs);
            if any(temp > 1e-13)
                disp([num2str(dt_obs), ' not compatible'])
                compatH(tt,nn) = 0;
            else
                disp([num2str(dt_obs), ' near compatible'])
                compatH(tt,nn) = 0.9;
            end
        end
    end
end
%%
figure(1); clf
str = cell(length(T_vals),1);
for i = 1:length(T_vals)
    loglog(dt_vals,fro(i,:)/norm(Q_fish,'fro')); hold on
    str{i} = ['T = ',num2str(T_vals(i))];
end
xlabel('$\Delta t_{obs}$','interpreter','latex','fontsize',16)
ylabel('$||\Delta t_{obs}H - Q_\infty||_F / ||Q_\infty||_F$','interpreter','latex','fontsize',16)
legend(str,'interpreter','latex','fontsize',14,'location','best'); legend boxoff
title('Fisher info: continuum limit','interpreter','latex','fontsize',18)