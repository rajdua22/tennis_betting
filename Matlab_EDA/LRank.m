%% Machine Learning Tennis Research Project
%
%  Rajvir Dua
%  ------------
% Simply predicts the better ranked player

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

p = X(:,3) < X(:,4);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

