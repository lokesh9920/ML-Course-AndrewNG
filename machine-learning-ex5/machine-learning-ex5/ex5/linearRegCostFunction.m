function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

sizeX = size(X); %12*2
sizeY = size(y); %12*1
sizeTheta = size(theta); %2*1

hypothesis = theta' * X';
hypothesis = hypothesis'; %12*1

diff = hypothesis-y;
squaredDiff = diff .^ 2;

squaredTheta = theta(2:end,:).^2; %not regularising theta0( i.e. 1)
J = sum(squaredDiff)/(2*m) + lambda * sum(squaredTheta)/(2*m);


%%%%%%
%%%%%%
%% IMPORTANT: IN grad(2:end) calculation for it to work for m =1 also, we need to take dimension parameter = 1 in (sum((hypothesis-y).*X,1)/m)', else 
% the learning curve submission will not work
% because for A = [ 1 2; 3 4]
% sum(A) gives column wise sum = [4 6]
% but for B = [1 2 3]
% sum(B) will not give column wise sum i.e [1 2 3], instead we get [6] since B is a vector. 
%%%%%%
%%%%%%
grad(1) = sum(hypothesis-y)/m;
grad(2:end) = ((sum((hypothesis-y).*X,1)/m)' + (theta*lambda)/m)(2:end);

% =========================================================================

grad = grad(:);

end
