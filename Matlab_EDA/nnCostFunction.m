function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
a1 = sigmoid(X * Theta1');
a1 = [ones(m,1) a1];
a2 = sigmoid(a1 * Theta2');
rsum  = 0;
for i = 1:m
    temp = a2(i, :);
    temp = temp';
    tempy = y(i);
    a = zeros(num_labels,1);
    a(tempy) = 1;
    b = (-a .* log(temp)) - ((1-a) .* log(1-temp));
    b = sum(b);
    rsum = rsum + b;
end
J = rsum / m;

Theta12  = Theta1(:, 2:input_layer_size+1).^2;
Theta22 = Theta2(:, 2:hidden_layer_size+1).^2;
c = 0;
for i=1:hidden_layer_size
    for j = 1:input_layer_size
        c = c + Theta12(i,j);
    end
end
for i = 1:num_labels
    for j = 1:hidden_layer_size
        c = c + Theta22(i,j);
    end
end
c = c * lambda;
c = c / 2;
c = c / m;
J = J + c;
    
% -------------------------------------------------------------

for i = 1:m
    a1 = X(i, :)';
    z2 = Theta1*a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    tempy = y(i);
    a = zeros(num_labels,1);
    a(tempy) = 1;
    d3 = a3 - a;
    d2 = (Theta2(:, 2:end)' * d3) .* sigmoidGradient(z2);
    Theta1_grad = (Theta1_grad + d2*a1');
    Theta2_grad = (Theta2_grad + d3*a2');
end
Theta1_grad = Theta1_grad ./m;
Theta2_grad = Theta2_grad ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
