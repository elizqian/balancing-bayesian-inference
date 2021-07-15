function nm = forstner(A,B)
    sig = eig(A,B);
    nm = sum(log(sig).^2);
end