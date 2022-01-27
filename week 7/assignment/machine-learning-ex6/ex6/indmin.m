function y = indmin(res)
%resturns the min index from the vector.
[~, rows] = min(res);
[~,cols] = min(min(res));
y = [ rows(cols),cols];
end