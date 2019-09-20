%% Machine Learning Tennis Research Project
%
%  Rajvir Dua
%  ------------
% Outdated - Don't use!!!!!
% USE LRWREG INSTEAD!

%% Initialization
clear ; close all; clc

%% Load Data
load('Data.mat');
Surface = Data(:,7);
Sets = Data(:,9);
Wrank = Data(:,12);
Lrank = Data(:,13);

Surface = table2array(Surface);
Surface = double(Surface);
Surface = Surface - 1;

Sets = table2array(Sets);
Sets = Sets == 3;

Wrank = table2array(Wrank);
% max2 = max(Wrank, [], 1, 'omitnan');
% mean2 = mean(Wrank, 'omitnan');
% Wrank = Wrank - mean2;
% Wrank = Wrank ./ max2;

Lrank = table2array(Lrank);
% max2 = max(Lrank, [], 1, 'omitnan');
% mean2 = mean(Lrank, 'omitnan');
% Lrank = Lrank - mean2;
% Lrank = Lrank ./ max2;

m = size(Wrank,1);
data = [Surface Sets Wrank Lrank ones(m,1)];
data = rmmissing(data);
X = data(:,1:4);
y = data(:,5);


mid = m / 2;
temp = X(mid:end,3);
X(mid:end, 3) = X(mid:end, 4);
X(mid:end, 4) = temp;
y(mid:end) = 0;



[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
% test_theta = [30; 0; 0.2; 0.2; 0.2];
% [cost, grad] = costFunction(test_theta, X, y);
% fprintf('\nCost at test theta: %f\n', cost);
% fprintf('Gradient at test theta: \n');
% fprintf(' %f \n', grad);


options = optimset('GradObj', 'on', 'MaxIter', 800);
 [theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);



prob = sigmoid([1 0 0 85 45] * theta);
fprintf(['For player1 with a rank of 45 and player2 with a rank of 85 (3 sets on carpet), we predict an win ' ...
         'probability of %f\n'], prob);
 p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);