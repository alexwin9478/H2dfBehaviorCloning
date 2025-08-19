function [output_ref, action_ref, imep_ref] = smoothened_steps(n_actions)
% imep_ref = 1e5 * [ones(1, 100) * 8, ones(1, 200) * 7, ones(1, 200) ...
%     * 4, ones(1, 200) * 3, ones(1, 100) * 5, ones(1, 200) * 6];
%load('nmpc_imep_ref_sequence.mat');


% IMEP, Nox, soot
% output_ref = [imep_ref(1:1000); zeros(1, 1000); zeros(1, 1000); zeros(1, 1000)];
% action_ref = zeros(n_actions, 1000);

length_ref = 4900;
load('smoothened_steps_550.mat');
imep_ref =  [smoothened_steps_550(1:length_ref)];
output_ref = [imep_ref(1:length_ref)*1e5; zeros(1, length_ref); zeros(1, length_ref); zeros(1, length_ref)];
action_ref = zeros(n_actions, length_ref);

end
