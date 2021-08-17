function check_compat(A,Gamma_pr, H)
temp = A*Gamma_pr + Gamma_pr*A';
eigpr = eig(temp);

if any(real(eigpr)>1e-9)
    disp('Prior not compatible, boo')
else
    disp('Prior compatible, yay')
end

temp = H*A + A'*H;
eigH = eig(temp);
if any(eigH>1e-13)
    disp('Fisher info not compatible, boo')
else
    disp('Fisher info compatible, yay')
end