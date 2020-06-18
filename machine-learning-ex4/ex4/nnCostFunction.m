function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%size(Theta1) 25 * 401
%size(Theta2) 10 * 26
%size (X) 5000 * 400

X = [ ones(size(X,1),1) X];
activated_layer2 = sigmoid( X * Theta1');

activated_layer2 = [ones(m,1) activated_layer2];
activated_layer3 = sigmoid(activated_layer2 * Theta2');

vectorized_y = zeros(num_labels,1);
vectorized_y(y(1))=1;
for i = 2:m
  temp = zeros(num_labels,1);
  temp(y(i))=1;
  vectorized_y = [ vectorized_y temp];
endfor;

J = sum(sum(vectorized_y'.*log(activated_layer3) + (1-vectorized_y').*log(1-activated_layer3)));
J = (-1)* J / m    + lambda * (sum(sum(Theta1(:,2:end) .* Theta1(:,2:end))) + sum(sum(Theta2(:,2:end) .* Theta2(:,2:end))))/(2 *m);

#---------------------------------------------------------------------------------------

Delta_3 = activated_layer3 - vectorized_y';
Delta_2 = Delta_3 * Theta2(:,2:end)  .* sigmoidGradient(X * Theta1');

% Dimestion of Delta_2 = 5000 * 26
% Dimension of Delta_3 = 5000 * 10
% Dimession of activated_layer2 = 5000 * 26

temp = Delta_2' * X;
Theta1_grad = (temp+ lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)])/m;
temp2 = Delta_3' * activated_layer2;

Theta2_grad = (temp2 + lambda *[zeros(size(Theta2,1),1) Theta2(:,2:end)]) /m;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
