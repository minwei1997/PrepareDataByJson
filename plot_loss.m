close all

% load loss data
data = csvread('loss.csv');

% draw all type loss
num_to_draw = 100;      % how many iteration need to draw

figure;
hold on
plot(data(1:num_to_draw,1));   % rpn_loss_cls
plot(data(1:num_to_draw,2));   % rpn_loss_box
plot(data(1:num_to_draw,3));   % loss_cls
plot(data(1:num_to_draw,4));   % loss_box
plot(data(1:num_to_draw,5));   % total_loss
legend('rpn loss cls', 'rpn loss box', 'loss cls', 'loss box', 'total loss')

% draw only total loss
figure;
plot(data(1:num_to_draw,5));   % total_loss
legend('total loss')