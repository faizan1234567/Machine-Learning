function J = CostFunction(X,y,Theta)
m = size(X,1);
prediction = X*Theta;
sqrError = (prediction -y).^2;
J = (1/(2*m) ) * sum(sqrError);
end
