clear all; close all;

%% target function
w = -1 + 2*rand(2, 1);

%% random samples & ground truth labels
samples = -1 + 2*rand(5000,2);
sumall = samples(:, 1) * w(1) + samples(:, 2) * w(2);
slabels = (sumall > 0).*2 - 1;

%% pla algorithm
weights = plaalg(samples, slabels);
