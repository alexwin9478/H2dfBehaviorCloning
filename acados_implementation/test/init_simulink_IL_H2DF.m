%% C2C NMPC
clear all;
% MPC struct
load('C2C_NMPC_struct.mat');
% MPC model and plant model struct
load('Par_2024_0008_0008_0271.mat');
H2DF_Par = Par;

% set reference: hard steps
[C2C_NMPC.outputs_ref, C2C_NMPC.controls_ref] = generate_reference_h2dual(C2C_NMPC.Dims.n_controls);
% simpler ref
imep_ref = step_ref_inverted(4);
C2C_NMPC.outputs_ref(1,:) = imep_ref(1,:);
C2C_NMPC.outputs_ref_norm = normalize_var(C2C_NMPC.outputs_ref, C2C_NMPC.Normalization.outputs.mean, C2C_NMPC.Normalization.outputs.std, 'to-scaled');
C2C_NMPC.controls_ref_norm = normalize_var(C2C_NMPC.controls_ref, C2C_NMPC.Normalization.controls.mean, C2C_NMPC.Normalization.controls.std, 'to-scaled');
C2C_NMPC.ref_norm = [C2C_NMPC.controls_ref_norm; C2C_NMPC.outputs_ref_norm; zeros(C2C_NMPC.Dims.n_controls, length(C2C_NMPC.controls_ref_norm(1, :)))];
C2C_NMPC.ref =[C2C_NMPC.controls_ref;C2C_NMPC.outputs_ref;zeros(C2C_NMPC.Dims.n_controls, length(C2C_NMPC.controls_ref(1, :)))];
C2C_NMPC.ref_norm_e = [C2C_NMPC.controls_ref_norm; C2C_NMPC.outputs_ref_norm];

% set reference: smoothened steps
IL = struct();
[IL.outputs_ref, IL.controls_ref] = generate_reference_h2dual(C2C_NMPC.Dims.n_controls);
%simpler ref
imep_ref_il= smoothened_steps(4);
IL.outputs_ref(1,:) = imep_ref_il(1,:);
IL.outputs_ref_norm = normalize_var(IL.outputs_ref, C2C_NMPC.Normalization.outputs.mean, C2C_NMPC.Normalization.outputs.std, 'to-scaled');
IL.controls_ref_norm = normalize_var(IL.controls_ref, C2C_NMPC.Normalization.controls.mean, C2C_NMPC.Normalization.controls.std, 'to-scaled');
IL.ref_norm = [IL.controls_ref_norm; IL.outputs_ref_norm; zeros(C2C_NMPC.Dims.n_controls, length(IL.controls_ref_norm(1, :)))];
IL.ref =[IL.controls_ref;IL.outputs_ref;zeros(C2C_NMPC.Dims.n_controls, length(IL.controls_ref(1, :)))];
IL.ref_norm_e = [IL.controls_ref_norm; IL.outputs_ref_norm];

%% Il Controller
IL_Data = struct2table(load('2024_545_to_550_IL_FF_normalized_post.mat'));

% IL data / post file, just for information
IL.Dims.n_inputs = 4;
IL.Labels.inputs = {'IMEP_ref_0', 'IMEP_ref_1', 'IMEP_ref_2', 'imep_old'}.';
IL.Units.inputs = {'pa', 'pa', 'pa', 'pa'};
IL.Dims.n_outputs = 4;
IL.Labels.outputs = {'doi_main', 'p2m', 'soi_main', 'doi_h2'}.';
IL.Units.outputs = {'s', 'us', '°CAbTDC', 's'};

IL_Data.label = string(IL_Data.label);
ind = boolean(sum(IL_Data.label == IL.Labels.outputs.', 2));
IL.Normalization.outputs.mean = [IL_Data.mean{boolean(ind)}].';
IL.Normalization.outputs.std = [IL_Data.std{ind}].';
ind = boolean(sum(IL_Data.label == IL.Labels.inputs.', 2));
IL.Normalization.inputs.mean = [IL_Data.mean{ind}].';
IL.Normalization.inputs.std = [IL_Data.std{ind}].';

for ii = 1:IL.Dims.n_inputs
    IL.Units.inputs_norm{ii} = [IL.Units.inputs{ii}, '/', num2str(IL.Normalization.inputs.std(ii))];
end
for ii = 1:IL.Dims.n_outputs
    IL.Units.outputs_norm{ii} = [IL.Units.outputs{ii}, '/', num2str(IL.Normalization.outputs.std(ii))];
end

C2C = Simulink.VariantConfigurationData;
C2C_tx = Simulink.VariantConfigurationData;