% sets up inference problem for the initial condition of the LTI system
% 
% this setup is shared between main and plotting scripts

%% load matrix operators for LIT model
model = 'ISS1R'; % heat, CD, beam, build, ISS1R

switch model
    case 'heat2'
        load('heat-cont.mat');
        d = size(A,1);
        B = eye(d);
        sig_obs = 0.008;
    case 'heat'
        load('heatmodel.mat')       % load LTI operators
        d = size(A,1);
        B = eye(d);                 % makes Pinf better conditioned than default B
        C = zeros(5,d);             % makes for slightly slower GEV decay than default C
        C(1:5,10:10:50) = eye(5);
    case 'CD'
        load('CDplayer.mat')
        d = size(A,1);
    case 'beam'
        load('beam.mat')
        d = size(A,1);
        B = eye(d);
    case 'ISS1R'
        load('iss1R.mat')
        d = size(A,1);
        sig_obs = [2.5e-3 5e-4 5e-4]';
    case 'build'
        load('build.mat')
        d = size(A,1);
end

d_out = size(C,1);

%% define inference problem
% define observation model (measurement times and noise scaling)
T       = 10;
dt_obs  = 1;       % making this bigger makes Spantini eigvals decay faster
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