function [ jval,gradient] = computeCostExp(theta)
%compute cost ans gradient and returns the cost functionb value and the
%gradient value
jval = (theta(1) -5)^2 + (theta(2) -5)^2;
gradient = zeros(2,1);
gradient(1) = 2*(theta(1) -5);
gradient(2) = 2*(theta(2) -5);
end


