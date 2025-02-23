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
squared_error_vector = ((X * theta)-y).*((X * theta)-y);
theta_squared = theta.*theta;
theta_squared(1)=0;
J = (sum(squared_error_vector) + lambda * sum ( theta_squared))/(2*m); 

%size(X)  12 * 2
%size(theta) 2* 1
%size(y) 12 * 1

grad = ((X'*((X * theta) - y)) + lambda * theta)/m;
grad(1,1) = (X(:,1:1)'*((X * theta) - y))/m;












% =========================================================================

grad = grad(:);

end
