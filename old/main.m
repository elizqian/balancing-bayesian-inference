clear; close all

% load LTI matrices, evolve true trajectory from true initial condition,
% generate data, plot outputs
setup;

% Spantini posterior mean and cov using exp G
r_vals = 1:20;
[mupos1, mupos2, Gpos,del2] = lowrankupdate(y,G,sig_obs,L_pr,r_vals);

sp_errs = zeros(length(r_vals),3);
for i = 1:length(r_vals)
    sp_errs(i,1) = norm(mupos1(:,i) - mupos1(:,end));
    sp_errs(i,2) = norm(mupos2(:,i) - mupos2(:,end));
    sp_errs(i,3) = norm(Gpos(:,:,i) - Gpos(:,:,end),'fro');
end

figure; clf
semilogy(del2,'ko'); hold on
semilogy(sp_errs)
xlim([0 r_vals(end)])
legend('Gen eigvals','class 1 mu err','class 2 mu err','Gpos err')
title('Spantini LR updates using exp G')

%% Balanced Truncation for Bayesian Inference
% use infinite noise obs Gram to define BT operators
Lf = lyapchol(A',C'/sig_obs)';
Qf = Lf*Lf';

% check difference between Fisher info and inf Gramian
dt_obs = obs_times(2) - obs_times(1);
norm(Qf - dt_obs*H,'fro')/norm(Qf,'fro')

[Ar,Br,Cr,Sr,Tr,hankel] = reduceLTI(max(r_vals),A,B,C,Lf,L_pr);
[Gr,~] = getGH(obs_times,Cr,Ar,sig_obs);

% plot BT outputs
figure; clf
yr = Gr*Sr'*x(:,1);
for i = 1:5
plot(obs_times,G(i:d_out:end,:)*x(:,1),'Color',getColor(i)); hold on
plot(obs_times,yr(i:d_out:end),'o','Color',getColor(i));
end
legend('Full model','BT','location','best')
title('BT vs full model outputs')

% use BT to compute posterior statistics
[muposBT, GposBT] = BTpos(y,Ar,Cr,Sr,sig_obs,Gamma_pr,obs_times,r_vals);
bt_errs = zeros(length(r_vals),2);
for i = 1:length(r_vals)
    bt_errs(i,1) = norm(muposBT(:,i) - mupos1(:,end));
    bt_errs(i,2) = norm(GposBT(:,:,i) - Gpos(:,:,end),'fro');
end

figure; clf
semilogy(hankel,'ko'); hold on
semilogy(bt_errs)
xlim([0 r_vals(end)])
legend('HSVs','mu err','Gpos err')
title('BT LR updates using exp G')


% BT Fisher mean and cov