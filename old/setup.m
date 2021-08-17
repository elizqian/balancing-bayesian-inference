load('heatmodel.mat')       % load LTI operators
d = size(A,1);
B = eye(d);                 % makes Pinf better conditioned than default B
C = zeros(5,197);           % makes for slightly slower GEV decay than default C
C(1:5,10:10:50) = eye(5);
d_out = size(C,1);

% define time for Euler
dt = 10;
T = 600;                
t = 0:dt:T;

%  define observation times and noise
num_obs = length(t)-1;
k = (length(t)-1)/num_obs;
obs_inds = k+1:k:length(t);
obs_times = t(k+1:k:end);
sig_obs = 0.04;

% compute compatible prior using A, B
L_pr = lyapchol(A,B)';
Gamma_pr = L_pr*L_pr';

% draw initial condition and evolve, plot
x0 = L_pr*randn(d,1);
x = backwardEuler(x0,t,A);

figure; 
plot(x(:,1)); hold on
for i = 1:num_obs
    plot(x(:,(i)*k+1),'Color',i/(num_obs+1)*[0.85 0.325 0.098]); hold on
end
title('Setup: true solution','fontsize',18)
legend('true x_0','later observation times','location','best','fontsize',16)

% plot true output and measured output at observation times
% get G_euler, G_exact
[G,H] = getGH(obs_times,C,A,sig_obs);
y = G*x(:,1) + sig_obs*randn(d_out*num_obs,1);
comp = G*x(:,1);
figure; 
for i = 1:5
plot(obs_times, C(i,:)*x(:,obs_inds),'Color',getColor(i)); hold on
plot(obs_times,y(i:d_out:end),'+','Color',getColor(i))
end
xlabel('t')
title('Setup: true and measured outputs','fontsize',18)
legend('True output','measured output','location','best')