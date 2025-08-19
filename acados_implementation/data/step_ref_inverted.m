function [output_ref, action_ref, imep_ref] = step_ref_inverted(n_actions)
% imep_ref = 1e5 * [ones(1, 100) * 8, ones(1, 200) * 7, ones(1, 200) ...
%     * 4, ones(1, 200) * 3, ones(1, 100) * 5, ones(1, 200) * 6];
%load('nmpc_imep_ref_sequence.mat');


% IMEP, Nox, soot
% output_ref = [imep_ref(1:1000); zeros(1, 1000); zeros(1, 1000); zeros(1, 1000)];
% action_ref = zeros(n_actions, 1000);

length_ref = 4900;
array_3 = ones(1, 500) * 3;
array_5 = ones(1, 600) * 5;
array_4 = ones(1,500)*4;
array_6 = ones(1,300)*6;
array_7 = ones(1,100)*6;
imep_ref =  [array_3, array_5,array_4,array_6, array_4, array_5, array_3, array_6,array_3,array_4, array_7];
output_ref = [imep_ref(1:length_ref)*1e5; zeros(1, length_ref); zeros(1, length_ref); zeros(1, length_ref)];
action_ref = zeros(n_actions, length_ref);

end
