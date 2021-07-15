function [G,H] = getGH(obs_times,C,A,sig_obs,dt)
if nargin == 4
%     disp('Calculating exact G')
    dt_obs = obs_times(2) - obs_times(1);
    iter = expm(A*dt_obs);
else
%     disp('Calculating Euler G')
    ImdtA = eye(size(A)) - dt*A;
    k = obs_times(1)/dt;
    iter = inv(ImdtA)^k;
end

n = length(obs_times);
[d_out,d] = size(C);
G = zeros(n*d_out,d);

temp = C;
for i = 1:n
    temp = temp*iter;
    G((i-1)*d_out+1:i*d_out,:) = temp;
end
H = G'*G/sig_obs^2;
