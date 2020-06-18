function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 10;
sigma = 3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


i=1;
C=[0.01,0.03,0.1,0.3,1,3,10,30];
sigma=[0.01,0.03,0.1,0.3,1,3,10,30];
index_min=1;

for index=1:length(C)
	
	for sigma_index=1:length(sigma)
		
		model=svmTrain(X,y,C(index),@(x1,x2)gaussianKernel(x1,x2,sigma(sigma_index)));
		predictions=svmPredict(model,Xval);
		error_(i)=mean(predictions~=yval);
		if(i!=1)
			if(error_(i)<error_(index_min))
				C_Sel=C(index);
				sigma_Sel=sigma(sigma_index);
				index_min=i;
			end       
		end
		i=i+1;
		
	endfor	
	
endfor


C=C_Sel;
sigma=sigma_Sel;

% =========================================================================

end

