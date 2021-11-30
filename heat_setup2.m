% sets up inference problem for the initial condition of the LTI system
% 
% this setup is shared between main and plotting scripts

%% load matrix operators for LIT model
model = 'heat2'; % heat, CD, beam, build, iss1R

load('models/heat-cont.mat');
d = size(A,1);
B = eye(d);
sig_obs = 0.008;

d_out = size(C,1);

%% define inference problem
% define observation model (measurement times and noise scaling)
T       = 10;
dt_obs  = 1e-1;       % making this bigger makes Spantini eigvals decay faster
n       = round(T/dt_obs);
obs_times = dt_obs:dt_obs:n*dt_obs;
scl_sig_obs = 0.1;   % relative noise scaling

% compute compatible prior
L_pr = lyapchol(A,B)';  
Gamma_pr = L_pr*L_pr';

% define full forward model 
G = zeros(n*d_out,d);
iter = expm(A*dt_obs);
temp = C;
for i = 1:n
    temp = temp*iter;
    G((i-1)*d_out+1:i*d_out,:) = temp;
end

% draw random IC and generate measurements
x0 = L_pr*randn(d,1);
y = G*x0;
if ~exist('sig_obs','var')
    sig_obs = scl_sig_obs*max(abs(reshape(y,d_out,n)),[],2);
end
sig_obs_long = repmat(sig_obs,n,1);
m = y + sig_obs_long.*randn(n*d_out,1);

F = C./sig_obs;
Go = G./sig_obs_long;

% compute Obs Gramian and Fisher info
L_Q = lyapchol(A',F')';
Q_inf = L_Q*L_Q';
H = Go'*Go;

figure(1); clf
for i = 1:d_out
    subplot(d_out,1,i)
    plot(obs_times,y(i:d_out:end),'Color',getColor(1),'linewidth',1); hold on
    plot(obs_times,m(i:d_out:end),'Color',[getColor(1) 0.1],'linewidth',2)
    ylabel(['output ',num2str(i)],'interpreter','latex')
end
legend({'True output','Measurements'},'interpreter','latex','location','southeast'); legend boxoff
xlabel('$t$','interpreter','latex')
sgtitle([model, ' model: true/measured outputs'],'interpreter','latex','FontSize',16)

lyapH = H*A+A'*H;
eigH = eig(lyapH);
if any(eigH>1e-13)
    disp('Fisher info not compatible, boo')
else
    disp('Fisher info compatible, yay')
end

figure(2);
plot(eigH,'o')
title('Eigenvalues of $HA + A^\top H$','interpreter','latex')