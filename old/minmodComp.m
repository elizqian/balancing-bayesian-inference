function R_Gam=minmodComp(Gamma0,A)
% Find a minimal modification of Gamma0 making it prior-compatible with A
% Input: Gamma0 - initial covariance matrix (n x n positive def matrix)
%         A - dynamic system matrix (n x n strictly stable matrix)
% Returns:  R_Gam - the Cholesky factor of a minimal modification of Gamma0 
%   making it compatible with A:  Gamma1 = R_Gam'*R_Gam = Gamma0 + E'*E  
%   such that  A*Gamma1+Gamma1*A' is the *nearest* negative semidefinite 
%   matrix to A*Gamma0+Gamma0*A' - BUT E'*E  is *not* the minimal 
%   perturbation to Gamma0 required for prior-compatibility.
% (Requires lyapchol from control system toolbox). 
% 
R0=chol(Gamma0); 
M0=A*Gamma0+Gamma0*A'; 
[V,D]=eig(M0); eigval=real(diag(D));
flagPos=(eigval>0); 
% If Gamma0 is prior-compatible then flagPos is empty
if isempty(flagPos)
    R_Gam=R0;
else
    d_pos=sqrt(eigval(flagPos));
    V_pos=V(:,flagPos);
    E = lyapchol(A,V_pos*diag(d_pos));
    [~,R_Gam]=qr([R0;E],0);
end