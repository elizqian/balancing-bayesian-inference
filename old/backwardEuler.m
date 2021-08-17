function x = backwardEuler(x0,t,A)

d = length(x0);
dt = t(2)-t(1);
ImdtA = eye(d) - dt*A;

x = zeros(d,length(t));
x(:,1) = x0;
for i = 1:length(t)-1
    x(:,i+1) = ImdtA\x(:,i);
end