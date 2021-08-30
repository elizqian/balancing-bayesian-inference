function nm = forstner(A,B,form)
if strcmp(form, 'sqrt')
    sig = gev_sqrt(A,B);
elseif strcmp(form, 'spd')
    sig = eig(A,B,'chol');
end
    nm = sum(log(sig).^2);
end