function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


initH = theta' * X';
initH = initH';

hyp = sigmoid(initH);

term1 = -y.* log(hyp);
term2 = (1-y).*log(1-hyp);

squaredTheta = theta.^2;
regularizationTermInCost = lambda * (sum(squaredTheta(2:length(squaredTheta))))/(2*m);

J = (sum(term1-term2))/m + regularizationTermInCost;

grad(1) = sum((hyp-y).*X(:,1))/m
for i = 2:length(grad)
  grad(i) = sum((hyp-y).*X(:,i))/m + (lambda * theta(i))/m;
end;



% =============================================================

end
