function savePDF(name,papersize,margins)
if nargin == 1
    papersize = [3.2 2.5];
    margins = [.1 .1];
end
paperposition = [margins, papersize(1)-2*margins(1), papersize(2)-2*margins(2)];
orient landscape
set(gcf,'papersize',papersize)
set(gcf,'paperposition',paperposition)
saveas(gcf, [name,'.pdf']);