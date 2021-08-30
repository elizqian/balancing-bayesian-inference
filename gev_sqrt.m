% function GEV_SQRT.m
% solves generalized eigenvalue problem LL'*v = lambda*RR'*v
% S = GEV_SQRT(L,R) returns only generalized eigenvalues
% [S,V] = GEV_SQRT(L,R) returns generalized eigenvalues and eigenvectors

function varargout = gev_sqrt(L,R)
    temp = L'/R';
    if nargout == 1
        varargout{1} = svd(temp).^2;
    elseif nargout == 2
        [~,S,V] = svd(temp);
        Vhat = R'\V;
        varargout{1} = diag(S).^2;
        varargout{2} = Vhat;
    else
        error('requested too many outputs in gev.m')
    end
end