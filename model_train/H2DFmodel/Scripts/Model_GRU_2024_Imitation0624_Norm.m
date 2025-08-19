%{ 
Authors:    Alexander Winkler(winkler_a@mmp.rwth-aachen.de)            
            Vasu Sharma(vasu3@ualberta.ca),

Copyright 2023 MECE,University of Alberta,
               Teaching and Research 
               Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
%}

clear
close all
clc
%% Set PWD & Path
% Windows Alex
% has to be executed in scripts folder, where this script here lies
addpath('../Functions/') %for mygrustateFnc etc.
addpath('D:/_Git/matlab2tikz/src') %for tikz (pull repo online by yourself: https://github.com/matlab2tikz/matlab2tikz)

%% Settings
do_training = true; 
no_p2m = true;
break_loop = false;
plot_vars = false; 
plot_vars_datasets = false;
plot_explainability = false;
plot_pred_val = false;
plot_pred_test = false;
plot_init = false; % plotting to investigate dataset
plot_init_measurements = false; % look on lvl 1 and lvl2 counters of the individual measurements
kill_violated_data = false; % kill lvl1 and lvl2 hits datapoints
kill_points_zero_h2_doi = false; % when safety lvl 2 was hit, standard controlelr was on until the end of the feq cycle, so kill these cycles here
save_plots_sw = false; 
save_analysis = true; 
plot_step_zoom = false;
plot_training_loss = false;
verify_my_func = false;
RNG = false;

% plotting options
Opts.multi_lines_ylabel = true;
Opts.LatexLabels = true;
Opts.fontsize = 11; % diss needs 13, should have no influence though
Opts.ltx_tma = '\tMa';
Opts.ltx_ptom = '\tPtoM';
Opts.ltx_ama = '\aMa';
Opts.ltx_thy = '\tHy';
Opts.ltx_pme = '\pMe';
Opts.ltx_cnox = '\cNox';
Opts.ltx_cpm = '\cPm';
Opts.ltx_dpm = '\dpm';
fntsze = Opts.fontsize;
tick_step = 500;

ratio_train = 0.8;
ratio_val = 0.95;

MP = 2024;
trainingrun = 298; no_fb = false; % best BC, with fb (normal data)
% 298: imitaition dataset, standard FF network, 12 units, FB, 0.007, 0.1184
% (15k learnables)


%% Load data
% call function
load('Test_545_NoPress.mat')
idx_start = 100;
idx_end = 29426;
[utrain_545, ytrain_545, uval_545, yval_545, utest_545, ytest_545] = getDatasets_IL(Test_545_NoPress, idx_start, idx_end, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi, RNG);

load('Test_547_NoPress.mat')
idx_start = 90;
idx_end = 28865;
[utrain_547, ytrain_547, uval_547, yval_547, utest_547, ytest_547] = getDatasets_IL(Test_547_NoPress, idx_start, idx_end, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi, RNG);

load('Test_550_NoPress.mat')
idx_start = 45;
idx_end = 27977;
[utrain_550, ytrain_550, uval_550, yval_550, utest_550, ytest_550] = getDatasets_IL(Test_550_NoPress, idx_start, idx_end, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi, RNG);

savename_data = '2024_545_to_550_IL_FF_normalized_post.mat';
savename_datasets = '2024_545_to_550_IL_FF_normalized_datasets.mat';
savename_datasets_phys = '2024_545_to_550_IL_FF_phys_datasets.mat';

%concatenate
utrain = [utrain_545'; utrain_547'; utrain_550'];
ytrain = [ytrain_545'; ytrain_547'; ytrain_550'];

uval = [uval_545'; uval_547'; uval_550'];
yval = [yval_545'; yval_547'; yval_550'];

utest = [utest_545'; utest_547'; utest_550'];
ytest = [ytest_545'; ytest_547'; ytest_550'];

utotal = [utrain; uval; utest];
ytotal = [ytrain; yval; ytest];

% 
% y1 = DOI_main_cycle';
% y2 = P2M_cycle';
% y3 = SOI_main_cycle';
% y4 = H2_doi_cycle';
% 
% u1 = IMEP_ref0_cycle';
% u2 = IMEP_ref1_cycle';
% u3 = IMEP_ref2_cycle';
% u4 = IMEP_old'; % feedback of old IMEP

% get data from concatenated datasets
DOI_main_cycle = ytotal(:,1)';
P2M_cycle = ytotal(:,2)';
SOI_main_cycle = ytotal(:,3)';
H2_doi_cycle = ytotal(:,4)'; % convert s to ms
IMEP_ref_0_cycle = utotal(:,1);
IMEP_ref_1_cycle = utotal(:,2);
IMEP_ref_2_cycle = utotal(:,3);
IMEP_old = utotal(:,4)';
% IMEP_cycle = ytotal(:,1)'; % pressure, bascically load in MPa
% NOx_cycle = ytotal(:,2)'; % cheap CAN sensor, not FTIR! in ppm
% Soot_cycle = ytotal(:,3)'; % in mgm3
% MPRR_cycle = ytotal(:,4)'; % dp max, in Mpa per CAD

%% mkdir
mkdir('../Plots/'+ sprintf("%04d",MP)',sprintf('%04d',trainingrun))
mkdir('../','/Results/')

max_scale = length(utotal); % 65024 for VSR1123002

%% if plot_vars (plot inputs, outputs, histogram whole dataset)
if plot_vars
%% ploting outputs whole dataset
figure
set(gcf, 'Position', [100, 100, 1800, 1000]);
set(gcf,'color','w');

ax5=subplot(4,1,1);
plot(DOI_main_cycle*1e3, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'Main Inj. DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        ylabel('Main Inj. DOI / ms','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax6=subplot(4,1,2);
plot(P2M_cycle, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ \\ / \mu s')},'Interpreter','latex')
    else
        ylabel({'P2M';'/ us'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        ylabel('P2M / us','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax7=subplot(4,1,3);
plot(SOI_main_cycle, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ \\ / CADbTDC')},'Interpreter','latex')
    else
        ylabel({'SOI Main';'/ CADbTDC'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        ylabel('SOI Main / CADbTDC','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax8=subplot(4,1,4);
plot(H2_doi_cycle*1e3, 'k','LineWidth',1)
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'H2 DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        ylabel('H2 DOI / ms','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Outputs"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end

%% ploting inputs whole dataset
figure
% set(gcf, 'Position', [100, 100, 1800, 800]);
set(gcf,'color','w');

%--------------------------------------------------
ax1=subplot(2,1,1);
plot(IMEP_ref_0_cycle*1e-5, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' Ref. (k)$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP Ref. (k)';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' Ref. (k)$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP Ref. (k) / bar','Interpreter','latex')
    end
end
% ylabel({'IMEP Ref k',' / bar'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
%--------------------------------------------------
ax2=subplot(2,1,2);
plot(IMEP_old*1e-5, 'k','LineWidth',1)
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' (k-1)$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP (k-1)';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' (k-1)$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP (k-1) / bar','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

linkaxes([ax1,ax2],'x');

set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Inputs"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end


%% common plot inputs outputs
figure
set(gcf, 'Position', [100, 100, 1800, 1000]);
set(gcf,'color','w');

%--------------------------------------------------
ax1=subplot(6,1,1);
plot(IMEP_ref_0_cycle*1e-5, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' Ref. (k)$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP Ref. (k)';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' Ref. (k)$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP Ref. (k) / bar','Interpreter','latex')
    end
end
% ylabel({'IMEP Ref k',' / bar'},'Interpreter','latex')
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
%--------------------------------------------------
ax2=subplot(6,1,2);
plot(IMEP_old*1e-5, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' (k-1)$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP (k-1)';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' (k-1)$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP (k-1) / bar','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax5=subplot(6,1,3);
plot(DOI_main_cycle*1e3, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'Main Inj. DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        ylabel('Main Inj. DOI / ms','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax6=subplot(6,1,4);
plot(P2M_cycle, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ \\ / \mu s')},'Interpreter','latex')
    else
        ylabel({'P2M';'/ us'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        ylabel('P2M / us','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax7=subplot(6,1,5);
plot(SOI_main_cycle, 'k','LineWidth',1)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ \\ / CADbTDC')},'Interpreter','latex')
    else
        ylabel({'SOI Main';'/ CADbTDC'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        ylabel('SOI Main / CADbTDC','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

ax8=subplot(6,1,6);
plot(H2_doi_cycle*1e3, 'k','LineWidth',1)
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'H2 DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        ylabel('H2 DOI / ms','Interpreter','latex')
    end
end
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/InputsOutputs"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end

%% ploting ref trajectory whole dataset
figure
% set(gcf, 'Position', [100, 100, 1800, 800]);
set(gcf,'color','w');

%--------------------------------------------------
plot(IMEP_ref_0_cycle*1e-5, 'k','LineWidth',1)
% plot(IMEP_ref_0_cycle(55940:58440)*1e-5, 'k','LineWidth',1)
grid on
xlabel("Cycles / -",'Interpreter', 'latex')

    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,' Ref.$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP Ref. / bar','Interpreter','latex')
    end

% ylabel({'IMEP Reference / bar'},'Interpreter','latex')
if plot_step_zoom
    xlim([56000,58500]);
else
    xlim([0,max_scale]);
end
ylim([3,8])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Ref_Time_Zoom"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end


%% Histogram on whole dataset outputs
fig = figure;
set(gcf,'color','w');

ax5=subplot(2,2,1);
histogram(DOI_main_cycle*1e3)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        xlabel('Main Inj. DOI / ms','Interpreter','latex')
    end
ylabel("Count / -",'Interpreter', 'latex')
% xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;


ax6=subplot(2,2,2);
histogram(P2M_cycle)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel("Count / -",'Interpreter', 'latex')
if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        xlabel('P2M / us','Interpreter','latex')
    end
% xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

ax7=subplot(2,2,3);
histogram(SOI_main_cycle)
grid on
ylabel("Count / -",'Interpreter', 'latex')
if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        xlabel('SOI Main / CADbTDC','Interpreter','latex')
    end
% xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

ax8=subplot(2,2,4);
histogram(H2_doi_cycle*1e3)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        xlabel('H2 DOI / ms','Interpreter','latex')
    end
% ylabel("Count / -",'Interpreter', 'latex')
% xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

% Give common xlabel, ylabel and title to your figure
% ylabel_position = [0.1, 0.5, 0, 0]; % x, y, z in Normalized Units
% annotation('textbox', ylabel_position, 'String', 'Count / -', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', fntsze, 'Interpreter','latex', 'EdgeColor', 'none', 'Rotation', 90);
% annotation('textbox', ylabel_position, 'String', 'Count / -', ...
    % 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    % 'FontSize', fntsze, 'EdgeColor', 'none', 'Rotation', 90);
set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Outputs_Histogram_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

%% Histogram on whole dataset
fig = figure;
subplot(2,1,1)
set(gcf,'color','w');
histogram(IMEP_old*1e-5)
grid on
if Opts.LatexLabels
    xlabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
else
    xlabel('IMEP / bar','Interpreter','latex')
end
ylabel("Count / -",'Interpreter', 'latex')
set(gca,'FontSize',fntsze)
% title('IMEP Data Distribution','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])


subplot(2,1,2)
set(gcf,'color','w');
histogram(IMEP_old*1e-5 - (IMEP_ref_0_cycle*1e-5)' )
grid on
% xlabel({'IMEP Tracking Error / bar'},'Interpreter','latex')
if Opts.LatexLabels
    xlabel({strcat('$',Opts.ltx_pme,' Tracking Error$ / bar')},'Interpreter','latex')
else
    xlabel('IMEP Tracking Error / bar','Interpreter','latex')
end
ylabel("Count / -",'Interpreter', 'latex')
set(gca,'FontSize',fntsze)
% title('IMEP Tracking Error Data Distribution','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Outputs_Data_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

end

%% analysis array init
runs_total_max = 30;
analysis = struct();
analysis.FinalRMSE = zeros(runs_total_max, 1);
analysis.FinalValidationLoss = zeros(runs_total_max, 1);
analysis.TotalLearnables = zeros(runs_total_max, 1);
analysis.ElapsedTime = zeros(runs_total_max, 1);
A = ['XXXX_00YY_00YY_0ZZZ.mat'];
analysis.savename = repmat(A, runs_total_max, 1);
run_nmbr = 0;


%% plot histograms and steps on the different datasets (train, val, test)
if plot_vars_datasets == true 

IMEP_old_cycle_tr = utrain(:,4);
IMEP_old_cycle_val = uval(:,4);
IMEP_old_cycle_test = utest(:,4);
max_scale = max( [max(utrain(:,4)), max(uval(:,4)), max(utest(:,4))]);
min_scale = min( [min(utrain(:,4)), min(uval(:,4)), min(utest(:,4))]);

% start plotting
% imep plots
fig = figure;
subplot(3,1,1)
set(gcf,'color','w');
histogram(IMEP_old_cycle_tr*1e-5)
grid on
xlim([min_scale*1e-5,max_scale*1e-5])
% xlabel({'IMEP [Pa]'},'Interpreter','latex')
% ylabel("Count / -",'Interpreter', 'latex')
% title('IMEP Training Data Distribution','Interpreter', 'latex')
title('Training Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca; ax.XTickLabel = [];

subplot(3,1,2)
set(gcf,'color','w');
histogram(IMEP_old_cycle_val*1e-5)
grid on
xlim([min_scale*1e-5,max_scale*1e-5])
% xlabel({'IMEP [Pa]'},'Interpreter','latex')
ylabel("Count / -",'Interpreter', 'latex')
% title('IMEP Validation Data Distribution','Interpreter', 'latex')
title('Validation Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax1 = gca; ax1.XTickLabel = [];

subplot(3,1,3)
set(gcf,'color','w');
histogram(IMEP_old_cycle_test*1e-5)
grid on
xlim([min_scale*1e-5,max_scale*1e-5])
% xlabel({'IMEP / bar'},'Interpreter','latex')
if Opts.LatexLabels
    xlabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
else
    xlabel('IMEP / bar','Interpreter','latex')
end
% ylabel("Count / -",'Interpreter', 'latex')
% title('IMEP Test Data Distribution','Interpreter', 'latex')
title('Test Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax2 = gca; % ax2.XTick = 0:1e5:11e5; % ax2.XRuler.Exponent = 0;
% sgt = sgtitle('IMEP Data Distribution','Interpreter', 'latex','Color','black', 'FontSize', fntsze);
linkaxes([ax ax1 ax2],'x')


if save_plots_sw
    type = "/IMEP_Data_Distribution_Sets"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

end
%% Normalizing data - ONLY with complete dataset!
% DOI_main_cycle = ytotal(:,1)';
% P2M_cycle = ytotal(:,2)';
% SOI_main_cycle = ytotal(:,3)';
% H2_doi_cycle = ytotal(:,4)'; % convert s to ms
% IMEP_ref_0_cycle = utotal(:,1);
% IMEP_ref_1_cycle = utotal(:,2);
% IMEP_ref_2_cycle = utotal(:,3);
% IMEP_old = utotal(:,4)';

[~, y1_min, y1_range] = dataTrainNormalize(DOI_main_cycle');
[~, y2_min, y2_range] = dataTrainNormalize(P2M_cycle');
[~, y3_min, y3_range] = dataTrainNormalize(SOI_main_cycle');
[~, y4_min, y4_range] = dataTrainNormalize(H2_doi_cycle');

[~, u1_min, u1_range] = dataTrainNormalize(IMEP_old');
[~, u2_min, u2_range] = dataTrainNormalize(IMEP_old');
[~, u3_min, u3_range] = dataTrainNormalize(IMEP_old');
[~, u4_min, u4_range] = dataTrainNormalize(IMEP_old');

utrain_1 = normalize_var(utrain(:,1), ...
    u1_min, u1_range, 'to-scaled');
utrain_2 = normalize_var(utrain(:,2), ...
    u2_min, u2_range, 'to-scaled');
utrain_3 = normalize_var(utrain(:,3), ...
    u3_min, u3_range, 'to-scaled');
utrain_4 = normalize_var(utrain(:,4), ...
    u4_min, u4_range, 'to-scaled');

ytrain_1 = normalize_var(ytrain(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytrain_2 = normalize_var(ytrain(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytrain_3 = normalize_var(ytrain(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytrain_4 = normalize_var(ytrain(:,4), ...
    y4_min, y4_range, 'to-scaled');

uval_1 = normalize_var(uval(:,1), ...
    u1_min, u1_range, 'to-scaled');
uval_2 = normalize_var(uval(:,2), ...
    u2_min, u2_range, 'to-scaled');
uval_3 = normalize_var(uval(:,3), ...
    u3_min, u3_range, 'to-scaled');
uval_4 = normalize_var(uval(:,4), ...
    u4_min, u4_range, 'to-scaled');

yval_1 = normalize_var(yval(:,1), ...
    y1_min, y1_range, 'to-scaled');
yval_2 = normalize_var(yval(:,2), ...
    y2_min, y2_range, 'to-scaled');
yval_3 = normalize_var(yval(:,3), ...
    y3_min, y3_range, 'to-scaled');
yval_4 = normalize_var(yval(:,4), ...
    y4_min, y4_range, 'to-scaled');

utest_1 = normalize_var(utest(:,1), ...
    u1_min, u1_range, 'to-scaled');
utest_2 = normalize_var(utest(:,2), ...
    u2_min, u2_range, 'to-scaled');
utest_3 = normalize_var(utest(:,3), ...
    u3_min, u3_range, 'to-scaled');
utest_4 = normalize_var(utest(:,4), ...
    u4_min, u4_range, 'to-scaled');

ytest_1 = normalize_var(ytest(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytest_2 = normalize_var(ytest(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytest_3 = normalize_var(ytest(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytest_4 = normalize_var(ytest(:,4), ...
    y4_min, y4_range, 'to-scaled');


%% Dateset Definition
% with IMEP feedback u5
if no_fb
    utrain_load = [utrain_1'; utrain_2'; utrain_3'];
    ytrain_load = [ytrain_1'; ytrain_2'; ytrain_3'; ytrain_4'];
    
    uval_load = [uval_1'; uval_2'; uval_3'];
    yval_load = [yval_1'; yval_2'; yval_3'; yval_4'];
    
    utest_load = [utest_1'; utest_2'; utest_3'];
    ytest_load = [ytest_1'; ytest_2'; ytest_3'; ytest_4'];
else
    utrain_load = [utrain_1'; utrain_2'; utrain_3'; utrain_4'];
    ytrain_load = [ytrain_1'; ytrain_2'; ytrain_3'; ytrain_4'];
    
    uval_load = [uval_1'; uval_2'; uval_3'; uval_4'];
    yval_load = [yval_1'; yval_2'; yval_3'; yval_4'];
    
    utest_load = [utest_1'; utest_2'; utest_3'; utest_4'];
    ytest_load = [ytest_1'; ytest_2'; ytest_3'; ytest_4'];
end

%% Save Datafiles
Data = struct();
% IMEP_ref_0_cycle = utotal(:,1);
% IMEP_ref_1_cycle = utotal(:,2);
% IMEP_ref_2_cycle = utotal(:,3);
% IMEP_old = utotal(:,4)';
Data.label = {'doi_main'; 'p2m'; 'soi_main'; 'doi_h2'; ...
     'IMEP_ref_0'; 'IMEP_ref_1'; 'IMEP_ref_2'; 'imep_old'};
data = {DOI_main_cycle', P2M_cycle', SOI_main_cycle', H2_doi_cycle', ...
    IMEP_ref_0_cycle, IMEP_ref_1_cycle, IMEP_ref_2_cycle, IMEP_old'};
for ii = 1:length(data)
    [Data.signal{ii, 1}, Data.mean{ii, 1}, Data.std{ii, 1}] = ...
        normalize_data(data{ii});
end
Data.mean{5, 1} = Data.mean{8, 1}; Data.std{5, 1} = Data.std{8, 1}; 
Data.mean{6, 1} = Data.mean{8, 1}; Data.std{6, 1} = Data.std{8, 1}; 
Data.mean{7, 1} = Data.mean{8, 1}; Data.std{7, 1} = Data.std{8, 1}; 
label = Data.label; mean = Data.mean; std = Data.std; signal = Data.signal;
save(fullfile(['../Results/',savename_data]), 'label', 'mean', 'std', 'signal');
clear mean; clear std; 

DataSets = struct();
DataSets.utrain_load = utrain_load;
DataSets.ytrain_load = ytrain_load;
DataSets.uval_load = uval_load;
DataSets.yval_load = yval_load;
DataSets.utest_load = utest_load;
DataSets.ytest_load = ytest_load;
DataSets.ratio_train = ratio_train;
DataSets.ratio_val = ratio_val;
save(fullfile(['../Results/',savename_datasets]), 'DataSets');

DataSetsPhys = struct();
DataSetsPhys.utrain = utrain;
DataSetsPhys.ytrain = ytrain;
DataSetsPhys.uval = uval;
DataSetsPhys.yval = yval;
DataSetsPhys.utest = utest;
DataSetsPhys.ytest = ytest;
DataSetsPhys.ratio_train = ratio_train;
DataSetsPhys.ratio_val = ratio_val;
DataSetsPhys.label = Data.label; DataSetsPhys.mean = Data.mean; DataSetsPhys.std = Data.std;
save(fullfile(['../Results/',savename_datasets_phys]), 'DataSetsPhys');


%% Training
for numHiddenUnits1 = [12] % for loop for grid searhc / to try out different units number within the FF layer IMPORTANT PARAMETER
for LSTMStateNum= [8] % [8,16] % [4,4,6,6,8,8] % for loop for grid searhc / to try out different units number within the recurrent layer IMPORTANT PARAMETER
tic

run_nmbr = run_nmbr + 1;
disp ( ['Measurement Point / Save File Number ', num2str(trainingrun)] );
disp ( ['Grid Search Number Iteration ', num2str(run_nmbr)] );

% mat = [u1,u2,u3,u4,u5];
% plotmatrix(mat)

%% Create Newtwork arch + setting / options
numResponses = 4; % y1 y2 y3 y4
if no_fb
    featureDimension = 3; % u1 u2 u3 % without feedback imep
else
    featureDimension = 4; % u1 u2 u3 u4 u5
end
maxEpochs = 5000; % IMPORTANT PARAMETER
miniBatchSize = 512; % IMPORTANT PARAMETER


% architecture
% Networklayer_h2df = [...
%     sequenceInputLayer(featureDimension)
%     fullyConnectedLayer(4*numHiddenUnits1)
%     reluLayer    
%     fullyConnectedLayer(4*numHiddenUnits1)
%     reluLayer
%     fullyConnectedLayer(8*numHiddenUnits1)
%     reluLayer
%     lstmLayer(LSTMStateNum,'OutputMode','sequence',InputWeightsInitializer='he',RecurrentWeightsInitializer='he')
%     fullyConnectedLayer(8*numHiddenUnits1)
%     reluLayer
%     fullyConnectedLayer(4*numHiddenUnits1)
%     reluLayer
%     fullyConnectedLayer(numResponses)
%     regressionLayer];

if do_training == true
    Networklayer_h2df = [...
        sequenceInputLayer(featureDimension)
        fullyConnectedLayer(numHiddenUnits1)
        reluLayer
        lstmLayer(LSTMStateNum,'OutputMode','sequence',InputWeightsInitializer='he',RecurrentWeightsInitializer='he')
        fullyConnectedLayer(32*numHiddenUnits1)
        reluLayer
        fullyConnectedLayer(numResponses)
        regressionLayer];

        % training options
    options_tr = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'GradientThreshold',1, ...
        'SequenceLength',8192,... % 'longest'
        'Shuffle','once', ...
        'Plots','training-progress',...    
        'VerboseFrequency',64,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',250,...
        'LearnRateDropFactor',0.75,...
        'L2Regularization',0.1,...
        'ValidationFrequency',10,...
        'InitialLearnRate', 0.0005,...
        'Verbose', false, ...
        'ExecutionEnvironment', 'cpu', ...
        'ValidationData',[{uval_load} {yval_load}],...
        'OutputNetwork','best-validation-loss');
end
%% training and Saving model data
savename = [sprintf('%04d',MP),'_',sprintf('%04d',numHiddenUnits1),'_',sprintf('%04d',LSTMStateNum),'_',sprintf('%04d',trainingrun),'.mat'];

if do_training == true
    tic
    [h2df_model, h2df_model_infor] = trainNetwork(utrain_load,ytrain_load,Networklayer_h2df,options_tr);
    toc
    ElapsedTime = toc;
    h2df_model_analysis = analyzeNetwork(h2df_model); % analysis including total number of learnable parameters
    h2df_model_infor.ElapsedTime = ElapsedTime;

    save(['../Results/h2df_model_',savename],"h2df_model")
    save(['../Results/h2df_model_info_',savename],"h2df_model_infor")
    save(['../Results/h2df_model_analysis_',savename],"h2df_model_analysis")
else
    load(['../Results/h2df_model_',savename])
    load(['../Results/h2df_model_info_',savename])
    % load(['../Results/h2df_model_analysis_',savename])
end

%% performance meta data for grid search etc
analysis.FinalRMSE(run_nmbr,1) = h2df_model_infor.FinalValidationRMSE;
analysis.FinalValidationLoss(run_nmbr,1)  = h2df_model_infor.FinalValidationLoss;
% analysis.TotalLearnables(run_nmbr,1) = h2df_model_analysis.TotalLearnables;
analysis.ElapsedTime(run_nmbr,1) = h2df_model_infor.ElapsedTime;
analysis.savename(run_nmbr,1:length(savename)) = savename;
% savename

%% Plot Training Results
if plot_training_loss
num_epoch = round(length(h2df_model_infor.TrainingLoss) / 10);
maxIter = length(h2df_model_infor.TrainingLoss);
val_freq = 10;
TrainLossX = linspace(1,maxIter, maxIter);
TrainLossY = h2df_model_infor.TrainingLoss;

% for i = 1 : num_epoch
%     % train loss is mean over the validation frquency (e.g. 10 values)
%     TrainLoss(1, i) = mean(h2df_model_infor.TrainingLoss(1+((i-1)*val_freq) : (val_freq-1)+((i-1)*val_freq)));
% end
% TrainLoss(1,2:num_epoch) = h2df_model_infor.TrainingLoss(10:10:(num_epoch*10)-10);
ValLossX = zeros(1,num_epoch);
ValLossX(1,1) = 1;
ValLossX(1,2:num_epoch) = ( val_freq:val_freq:(num_epoch*val_freq)-val_freq );
ValLossY = zeros(1,num_epoch);
ValLossY(1,1) = h2df_model_infor.ValidationLoss(1);
ValLossY(1,2:num_epoch) = h2df_model_infor.ValidationLoss(1,val_freq:val_freq:(num_epoch*val_freq)-val_freq);

figure
set(gcf,'color','w');
set(gcf,'units','points','position',[200,200,900,400])
% plot(h2df_model_infor.TrainingLoss,"--", 'Color', 'blue','LineWidth',1);
plot(TrainLossX, TrainLossY,"--", 'Color', 'blue','LineWidth',1);
hold on
% plot(fillmissing(h2df_model_infor.ValidationLoss,'linear'), 'k','LineWidth',2);
plot(ValLossX, ValLossY, 'k','LineWidth',2);
% best validation loss, mark the epoch with the minimum validation loss and mark it with a cross
plot((h2df_model_infor.OutputNetworkIteration), h2df_model_infor.FinalValidationLoss, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
grid on
xlabel("Epochs / -",'Interpreter', 'latex');
ylabel("Loss / -",'Interpreter', 'latex');
legend("Training","Validation","Best Validation Loss - Final Network",'Location','northeast','Orientation','horizontal','Interpreter', 'latex');
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca, 'YScale', 'log')
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

% num_epoch = round(length(h2df_model_infor.TrainingLoss) / 10);
% val_freq = 10;
% TrainLoss = zeros(1, num_epoch);
% for i = 1 : num_epoch
%     % train loss is mean over the validation frquency (e.g. 10 values)
%     TrainLoss(1, i) = mean(h2df_model_infor.TrainingLoss(1+((i-1)*val_freq) : (val_freq-1)+((i-1)*val_freq)));
% end
% % TrainLoss(1,2:num_epoch) = h2df_model_infor.TrainingLoss(10:10:(num_epoch*10)-10);
% ValLoss = zeros(1,num_epoch);
% ValLoss(1,1) = h2df_model_infor.ValidationLoss(1);
% ValLoss(1,2:num_epoch) = h2df_model_infor.ValidationLoss(1,val_freq:val_freq:(num_epoch*val_freq)-val_freq);
% 
% figure
% set(gcf,'color','w');
% set(gcf,'units','points','position',[200,200,900,400])
% % plot(h2df_model_infor.TrainingLoss,"--", 'Color', 'blue','LineWidth',1);
% plot(TrainLoss,"--", 'Color', 'blue','LineWidth',1);
% hold on
% % plot(fillmissing(h2df_model_infor.ValidationLoss,'linear'), 'k','LineWidth',2);
% plot(ValLoss, 'k','LineWidth',2);
% % best validation loss, mark the epoch with the minimum validation loss and mark it with a cross
% plot((h2df_model_infor.OutputNetworkIteration / val_freq), h2df_model_infor.FinalValidationLoss, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
% grid on
% xlabel("Epochs / -",'Interpreter', 'latex');
% ylabel("Loss / -",'Interpreter', 'latex');
% legend("Training","Validation","Best Validation Loss - Final Network",'Location','northeast','Orientation','horizontal','Interpreter', 'latex');
% set(gcf,'units','points','position',[200,200,900,400])
% set(gca,'FontSize',fntsze)
% set(gca, 'YScale', 'log')
% set(gca,'TickLabelInterpreter','latex')
% ax = gca;
% ax.XRuler.Exponent = 0;

if save_plots_sw
    type = "/Loss_Iteration";
    resolution = 10;
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, resolution)
end

end % plot training loss 

%% Prediction on val dataset
if no_fb
    y_hat_val = predict(h2df_model,[uval_1'; uval_2'; uval_3']) ; % with IMEP
else
    y_hat_val = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4']) ; % with IMEP
end
% feedback
% y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4']) ;
y1_hat_val = y_hat_val(1,:);
y2_hat_val = y_hat_val(2,:);
y3_hat_val = y_hat_val(3,:);
y4_hat_val = y_hat_val(4,:);

% Denormalize Predictions
% [~, y1_min, y1_range] = dataTrainNormalize(DOI_main_cycle');
% [~, y2_min, y2_range] = dataTrainNormalize(P2M_cycle');
% [~, y3_min, y3_range] = dataTrainNormalize(SOI_main_cycle');
% [~, y4_min, y4_range] = dataTrainNormalize(H2_doi_cycle');
% [~, u1_min, u1_range] = dataTrainNormalize(IMEP_old');

DOI_main_cycle_hat_val = dataTraindeNormalize(y1_hat_val,y1_min,y1_range);
P2M_cycle_hat_val = dataTraindeNormalize(y2_hat_val,y2_min,y2_range);
SOI_main_cycle_hat_val = dataTraindeNormalize(y3_hat_val,y3_min,y3_range);
DOI_H2_cycle_hat_val = dataTraindeNormalize(y4_hat_val,y4_min,y4_range);

%% Explainability on val dataset
if plot_explainability
    if no_fb
        act = activations(h2df_model,[uval_1'; uval_2'; uval_3'],"gru");
    else
        act = activations(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4'],"gru");
    end
    heatmap(act{1,1}(1:8,1:100));
    xlabel("Cycle")
    ylabel("Hidden Unit")
    title("GRU Activations")
end

%% get rmses etc postprocessing VAL
% errperf(T,P,'mae')
% mse (mean squared error)
% rmse (root mean squared error)
% mspe (mean squared percentage error)
% rmspe (root mean squared percentage error)
%%: beware of conversion and units here! s vs ms for doi main and h2
%%doi!!!!
% prompt: 
% fill this latex table below with the correct values from the second input - the data. Check for the test dataset data (_test). 
% But please multiple MAE, RMSE from DOIm and DOIh2 by 1e3. Multiply the MSE value by 1e6. do not change the r2 and the NRMSE value. 
% Use the e-3 notation instead of the x 10^-3 notation.
% round to two decimal places for all values but for R square / R2, round to four there. 
% do not omit the percentage sign for nrmse. 
% Create a second table for the validation dataset (_val).
% 
% 
% 
% \begin{table}%[h!]
% 	\centering
% 	\begin{tabular}{|l|c|c|c|c} 
% 		\hline
% 		\textbf{Metric} &  \textbf{$\tMa$} &  \textbf{$\tPtoM$} &  \textbf{$\aMa$} &  \textbf{$\tHy$} \\
% 		\hline \hline
% 		MAE /  & 7.23 e-3 ms &  us & CAD &  ms \\
% 		\hline
% 		MSE /  &  ms$^2$ & us$^2$ & CAD$^2$ & ms$^2$ \\
% 		\hline
% 		RMSE /  & ms & us & CAD & ms \\
% 		\hline
% 		NRMSE / - & 2.\% & 9.\% & \% & \% \\
% 		\hline
% 		R2 / - & 0.9841 & 0.5617 \\
% 		\hline             
% 	\end{tabular}
% 	\caption[Behavior Cloning deep neural network prediction performance metrics on unseen test dataset.]{Behavior Cloning deep neural network prediction performance metrics on unseen test dataset. MAE: Mean Absolute Error, MSE: Mean Squared Error, RMSE: Root MSE, NRMSE: Normalized RMSE.}
% 	\label{tab:bc_train_metrics_test}
% \end{table}
% 
%  % get from excel copy paste and transpose from the matlab var
% analysis.maeDOIm_val	7,24E-06
%  ..
%  analysis.r2DOIh2_tst	0,962415516

% s to ms
analysis.target_range_DOIm = max(1e3*ytrain(:,1)) - min(1e3*ytrain(:,1));
analysis.std_DOIm = std(1e3*ytrain(:,1));
analysis.target_range_DOIm_sd4 = 4*std(1e3*ytrain(:,1));
analysis.target_range_DOIm_sd6 = 6*std(1e3*ytrain(:,1));
lower_p = prctile(1e3*ytrain(:,1), 1); upper_p = prctile(1e3*ytrain(:,1), 99);
analysis.range_clipped_DOIm = upper_p - lower_p;

analysis.target_range_P2M       = max(ytrain(:,2)) - min(ytrain(:,2));
analysis.std_P2M                = std(ytrain(:,2)');
analysis.target_range_P2M_sd4   = 4 * analysis.std_P2M;
analysis.target_range_P2M_sd6   = 6 * analysis.std_P2M;
low_p  = prctile(ytrain(:,2), 1); 
high_p = prctile(ytrain(:,2), 99);
analysis.range_clipped_P2M      = high_p - low_p;

analysis.target_range_SOIm       = max(ytrain(:,3)) - min(ytrain(:,3));
analysis.std_SOIm                = std(ytrain(:,3));
analysis.target_range_SOIm_sd4   = 4 * analysis.std_SOIm;
analysis.target_range_SOIm_sd6   = 6 * analysis.std_SOIm;
low_p  = prctile(ytrain(:,3), 1); 
high_p = prctile(ytrain(:,3), 99);
analysis.range_clipped_SOIm      = high_p - low_p;

analysis.target_range_DOIh2       = max(1e3*ytrain(:,4)) - min(1e3*ytrain(:,4));
analysis.std_DOIh2                = std(1e3*ytrain(:,4));
analysis.target_range_DOIh2_sd4   = 4 * analysis.std_DOIh2;
analysis.target_range_DOIh2_sd6   = 6 * analysis.std_DOIh2;
low_p  = prctile(1e3*ytrain(:,4), 1); 
high_p = prctile(1e3*ytrain(:,4), 99);
analysis.range_clipped_DOIh2      = high_p - low_p;

analysis.maeDOIm_val = errperf(1e3*yval(:,1)',1e3*DOI_main_cycle_hat_val, 'mae');
analysis.mseDOIm_val = errperf(1e3*yval(:,1)',1e3*DOI_main_cycle_hat_val, 'mse');
analysis.rmseDOIm_val = errperf(1e3*yval(:,1)',1e3*DOI_main_cycle_hat_val, 'rmse');
analysis.mspeDOIm_val = errperf(1e3*yval(:,1)',1e3*DOI_main_cycle_hat_val, 'mspe');
analysis.rmspeDOIm_val = errperf(1e3*yval(:,1)',1e3*DOI_main_cycle_hat_val, 'rmspe');
analysis.nrmseDOIm_val = ((analysis.rmseDOIm_val / analysis.target_range_DOIm)) * 100;

analysis.maeP2M_val = errperf(yval(:,2)',P2M_cycle_hat_val, 'mae');
analysis.mseP2M_val = errperf(yval(:,2)',P2M_cycle_hat_val, 'mse');
analysis.rmseP2M_val = errperf(yval(:,2)',P2M_cycle_hat_val, 'rmse');
analysis.mspeP2M_val = errperf(yval(:,2)',P2M_cycle_hat_val, 'mspe');
analysis.rmspeP2M_val = errperf(yval(:,2)',P2M_cycle_hat_val, 'rmspe');
analysis.nrmseP2M_val = ((analysis.rmseP2M_val / analysis.target_range_P2M)) * 100;

% SOI_main_cycle_hat_val = dataTraindeNormalize(y3_hat_val,y3_min,y3_range);
% y3_hat_val,y3_min,y3_range);
% yval_3 = normalize_var(yval(:,3), ...
%     y3_min, y3_range, 'to-scaled');
% analysis.rmspeSOIm_val = errperf(y3_hat_val', yval_3, 'rmspe');
% analysis.rmspeSOIm_val_denorm = dataTraindeNormalize(analysis.rmspeSOIm_val, y3_min, y3_range);

analysis.maeSOIm_val = errperf(yval(:,3)',SOI_main_cycle_hat_val, 'mae');
analysis.mseSOIm_val = errperf(yval(:,3)',SOI_main_cycle_hat_val, 'mse');
analysis.rmseSOIm_val = errperf(yval(:,3)',SOI_main_cycle_hat_val, 'rmse');
% analysis.mspeSOIm_val = errperf(yval(:,3)',SOI_main_cycle_hat_val, 'mspe');
% analysis.rmspeSOIm_val = errperf(yval(:,3)',SOI_main_cycle_hat_val, 'rmspe');
analysis.nrmseSOIm_val = ((analysis.rmseSOIm_val / analysis.target_range_SOIm)) * 100;

analysis.maeDOIh2_val = errperf(1e3*yval(:,4)',1e3*DOI_H2_cycle_hat_val, 'mae');
analysis.mseDOIh2_val = errperf(1e3*yval(:,4)',1e3*DOI_H2_cycle_hat_val, 'mse');
analysis.rmseDOIh2_val = errperf(1e3*yval(:,4)',1e3*DOI_H2_cycle_hat_val, 'rmse');
analysis.mspeDOIh2_val = errperf(1e3*yval(:,4)',1e3*DOI_H2_cycle_hat_val, 'mspe');
analysis.rmspeDOIh2_val = errperf(1e3*yval(:,4)',1e3*DOI_H2_cycle_hat_val, 'rmspe');
analysis.nrmseDOIh2_val = ((analysis.rmseDOIh2_val / analysis.target_range_DOIh2)) * 100;

% rmseDOIm_val = rmse((yval(:,1))',(DOI_main_cycle_hat_val),"all"); % bar
% target_range_DOIm_val = max(yval(:,1)) - min(yval(:,1));
% rmspeDOIm_val = ((rmseDOIm_val / target_range_DOIm_val)) * 100;
% rmseP2M_val=rmse(yval(:,2)',P2M_cycle_hat_val,"all");
% target_range_P2M_val = max(yval(:,2)) - min(yval(:,2));
% rmspeP2M_val = ((rmseP2M_val / target_range_P2M_val)) * 100;
% rmseSOIm_val=rmse(yval(:,3)', SOI_main_cycle_hat_val,"all");
% target_range_SOIm_val = max(yval(:,3)) - min(yval(:,3));
% rmspeSOIm_val = ((rmseSOIm_val / target_range_SOIm_val)) * 100;
% rmseDOIh2_val=rmse((yval(:,4)),DOI_H2_cycle_hat_val,"all");
% target_range_DOIh2_val = max((yval(:,4))) - min((yval(:,4)));
% rmspeDOIh2_val = ((rmseDOIh2_val / target_range_DOIh2_val)) * 100;

SSR_DOIm_val = sum((yval(:,1)' - DOI_main_cycle_hat_val).^2); % Sum of squared residuals
TSS_DOIm_val = sum(((DOI_main_cycle_hat_val - mean(DOI_main_cycle_hat_val)).^2)); % Total sum of squares
analysis.r2DOIm_val = 1 - SSR_DOIm_val/TSS_DOIm_val; % R squared
SSR_P2M_val = sum((yval(:,2)' - P2M_cycle_hat_val).^2); 
TSS_P2M_val = sum(((P2M_cycle_hat_val - mean(P2M_cycle_hat_val)).^2));
analysis.r2P2M_val = 1 - SSR_P2M_val/TSS_P2M_val;
SSR_SOIm_val = sum((yval(:,3)' - SOI_main_cycle_hat_val).^2); 
TSS_SOIm_val = sum(((SOI_main_cycle_hat_val - mean(SOI_main_cycle_hat_val)).^2));
analysis.r2SOIm_val = 1 - SSR_SOIm_val/TSS_SOIm_val;
SSR_DOIh2_val = sum((yval(:,4)' - DOI_H2_cycle_hat_val).^2); 
TSS_DOIh2_val = sum(((DOI_H2_cycle_hat_val - mean(DOI_H2_cycle_hat_val)).^2));
analysis.r2DOIh2_val = 1 - SSR_DOIh2_val/TSS_DOIh2_val;

%% Scatter plot measured vs. predicted (linear ideally) VAL
if plot_pred_val
figure
set(gcf,'color','w');
set(gcf,'units','points','position',[200,200,900,900])

subplot(2,2,1)
scatter(yval(:,1)*1e3, DOI_main_cycle_hat_val*1e3, 4, 'k', 'x', 'LineWidth', 5);
max_scale_x = round(max(DOI_main_cycle_hat_val*1e3), 5);
max_scale_y = round(max(yval(:,1)*1e3), 5);
min_scale_x = round(min(DOI_main_cycle_hat_val*1e3), 5);
min_scale_y = round(min(yval(:,1)*1e3), 5);
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')        
    else
        xlabel({'True DOI Main / ms'},'Interpreter','latex')
        ylabel({'Predicted DOI Main / ms'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.5f ms', analysis.rmseDOIm_val*1e3); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.5f', analysis.r2DOIm_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,2)
scatter(yval(:,2), P2M_cycle_hat_val, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(P2M_cycle_hat_val), -1);
max_scale_y = round(max(yval(:,2)), -1);
min_scale_x = round(min(P2M_cycle_hat_val), -1);
min_scale_y = round(min(yval(:,2)), -1);
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
else
        xlabel({'True P2M / us'},'Interpreter','latex')
        ylabel({'Predicted P2M / us'},'Interpreter','latex')
end

set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.1f us', analysis.rmseP2M_val); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.1f', analysis.r2P2M_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,3)
scatter(yval(:,3), SOI_main_cycle_hat_val, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(SOI_main_cycle_hat_val), 1);
max_scale_y = round(max(yval(:,3)), 1);
min_scale_x = round(min(SOI_main_cycle_hat_val), 1);
min_scale_y = round(min(yval(:,3)), 1);
% acc = 0.5; max_scale_x = round(max_scale_x/acc)*acc - 0.5;
% max_scale_y = round(max_scale_y/acc)*acc - 0.5; % MANIP
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        xlabel({'True SOI Main / CADbTDC'},'Interpreter','latex')
        ylabel({'Predicted SOI Main / CADbTDC'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f (bTDC CAD)', analysis.rmseSOIm_val); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2SOIm_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');


subplot(2,2,4)
scatter(yval(:,4)*1e3, DOI_H2_cycle_hat_val*1e3, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(DOI_H2_cycle_hat_val*1e3), 4);
max_scale_y = round(max(yval(:,4)*1e3), 4);
min_scale_x = round(min(DOI_H2_cycle_hat_val*1e3), 4);
min_scale_y = round(min(yval(:,4)*1e3), 4);
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        xlabel({'True DOI H2 / ms'},'Interpreter','latex')
        ylabel({'Predicted DOI H2 / ms'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.4f ms', analysis.rmseDOIh2_val*1e3); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.4f', analysis.r2DOIh2_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

if save_plots_sw
    type = "/Prediction_Actual_Val"; 
    resolution = 0;
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, resolution);
end


%% plotting data on val dataset TIME

figure
set(gcf,'color','w');
% set(gcf,'units','points','position',[200,200,900,400]) % paper
set(gcf,'units','points','position',[2030,193,780,600]) % powerpoint 50% 

subplot(4,1,1)
plot(yval(:,1)*1e3', 'r--')
hold on
plot(DOI_main_cycle_hat_val(1:end)*1e3,'k-')
grid on
% title('Prediction on Validation Data','Interpreter', 'latex')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'Main Inj. DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        ylabel('Main Inj. DOI / ms','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
% ax.XTick = 0:10000:90000;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')

subplot(4,1,2)
set(gcf,'units','points','position',[200,200,900,400])
plot(yval(:,2)', 'r--')
hold on
plot(P2M_cycle_hat_val(1:end),'k-')         
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ \\ / \mu s')},'Interpreter','latex')
    else
        ylabel({'P2M';'/ us'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        ylabel('P2M / us','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
% ax.XTick = 0:10000:90000;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')


subplot(4,1,3)
set(gcf,'units','points','position',[200,200,900,400])
plot(yval(:,3)', 'r--')
grid on
hold on
plot(SOI_main_cycle_hat_val(1:end),'k-')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ \\ / CADbTDC')},'Interpreter','latex')
    else
        ylabel({'SOI Main';'/ CADbTDC'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        ylabel('SOI Main / CADbTDC','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
% ax.XTick = 0:10000:90000;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')

subplot(4,1,4)
set(gcf,'units','points','position',[200,200,900,400])
plot(yval(:,4)*1e3', 'r--')
hold on
plot(DOI_H2_cycle_hat_val(1:end)*1e3,'k-')
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'H2 DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        ylabel('H2 DOI / ms','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
% ax.XTick = 0:10000:90000;
ax.XRuler.Exponent = 0;
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')

if save_plots_sw
    type = "/Prediction_Time_Val";
    resolution = 0;
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, resolution, Opts.multi_lines_ylabel)
end

end


%% Prediction on test dataset
if no_fb
    y_hat_tst = predict(h2df_model,[utest_1'; utest_2'; utest_3']) ; % with IMEP
else
    y_hat_tst = predict(h2df_model,[utest_1'; utest_2'; utest_3'; utest_4']) ; % with IMEP
end
% feedback
% y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4']) ;
y1_hat_tst = y_hat_tst(1,:);
y2_hat_tst = y_hat_tst(2,:);
y3_hat_tst = y_hat_tst(3,:);
y4_hat_tst = y_hat_tst(4,:);

% Denormalize Predictions
DOI_main_cycle_hat_tst = dataTraindeNormalize(y1_hat_tst,y1_min,y1_range);
P2M_cycle_hat_tst = dataTraindeNormalize(y2_hat_tst,y2_min,y2_range);
SOI_main_cycle_hat_tst = dataTraindeNormalize(y3_hat_tst,y3_min,y3_range);
DOI_H2_cycle_hat_tst = dataTraindeNormalize(y4_hat_tst,y4_min,y4_range);

%% Explainability on test dataset
if plot_explainability
    if no_fb
        act = activations(h2df_model,[utest_1'; utest_2'; utest_3'],"gru");
    else
        act = activations(h2df_model,[utest_1'; utest_2'; utest_3'; utest_4'],"gru");
    end
    heatmap(act{1,1}(1:8,1:100));
    xlabel("Cycle")
    ylabel("Hidden Unit")
    title("GRU Activations")
end

%% get rmses etc postprocessing TEST



% === IMEP (scaled) ===

% errperf(T,P,'mae')
% mse (mean squared error)
% rmse (root mean squared error)
% mspe (mean squared percentage error)
% rmspe (root mean squared percentage error)
analysis.maeDOIm_tst = errperf(1e3*ytest(:,1)',1e3*DOI_main_cycle_hat_tst, 'mae');
analysis.mseDOIm_tst = errperf(1e3*ytest(:,1)',1e3*DOI_main_cycle_hat_tst, 'mse');
analysis.rmseDOIm_tst = errperf(1e3*ytest(:,1)',1e3*DOI_main_cycle_hat_tst, 'rmse');
analysis.mspeDOIm_tst = errperf(1e3*ytest(:,1)',1e3*DOI_main_cycle_hat_tst, 'mspe');
analysis.rmspeDOIm_tst = errperf(1e3*ytest(:,1)',1e3*DOI_main_cycle_hat_tst, 'rmspe');
analysis.nrmseDOIm_tst = ((analysis.rmseDOIm_tst / analysis.target_range_DOIm)) * 100;

analysis.nrmse_sd6_DOIm_tst = analysis.rmseDOIm_tst / (6*analysis.std_DOIm) * 100;
analysis.nrmse_sd4_DOIm_tst = analysis.rmseDOIm_tst / (4*analysis.std_DOIm) * 100;
analysis.nrmse_clipped_DOIm_tst = analysis.rmseDOIm_tst / analysis.range_clipped_DOIm * 100;
analysis.nmae_DOIm_tst = (analysis.maeDOIm_tst / analysis.target_range_DOIm) * 100;
analysis.nmae_sd4_DOIm_tst = analysis.maeDOIm_tst / (4 * analysis.std_DOIm) * 100;
analysis.nmae_sd6_DOIm_tst = analysis.maeDOIm_tst / (6 * analysis.std_DOIm) * 100;
analysis.nmae_clipped_DOIm_tst = analysis.maeDOIm_tst / analysis.range_clipped_DOIm * 100;


analysis.maeP2M_tst = errperf(ytest(:,2)',P2M_cycle_hat_tst, 'mae');
analysis.mseP2M_tst = errperf(ytest(:,2)',P2M_cycle_hat_tst, 'mse');
analysis.rmseP2M_tst = errperf(ytest(:,2)',P2M_cycle_hat_tst, 'rmse');
analysis.mspeP2M_tst = errperf(ytest(:,2)',P2M_cycle_hat_tst, 'mspe');
analysis.rmspeP2M_tst = errperf(ytest(:,2)',P2M_cycle_hat_tst, 'rmspe');
% analysis.target_range_P2M_tst = max(ytrain(:,2)) - min(ytrain(:,2));
analysis.nrmseP2M_tst = ((analysis.rmseP2M_tst / analysis.target_range_P2M)) * 100;

analysis.nrmse_sd6_P2M_tst      = analysis.rmseP2M_tst   / (6 * analysis.std_P2M)   * 100;
analysis.nrmse_sd4_P2M_tst      = analysis.rmseP2M_tst   / (4 * analysis.std_P2M)   * 100;
analysis.nrmse_clipped_P2M_tst  = analysis.rmseP2M_tst   / analysis.range_clipped_P2M * 100;
analysis.nmae_P2M_tst           = analysis.maeP2M_tst    / analysis.target_range_P2M  * 100;
analysis.nmae_sd4_P2M_tst       = analysis.maeP2M_tst    / (4 * analysis.std_P2M)     * 100;
analysis.nmae_sd6_P2M_tst       = analysis.maeP2M_tst    / (6 * analysis.std_P2M)     * 100;
analysis.nmae_clipped_P2M_tst   = analysis.maeP2M_tst    / analysis.range_clipped_P2M * 100;


analysis.maeSOIm_tst = errperf(ytest(:,3)',SOI_main_cycle_hat_tst, 'mae');
analysis.mseSOIm_tst = errperf(ytest(:,3)',SOI_main_cycle_hat_tst, 'mse');
analysis.rmseSOIm_tst = errperf(ytest(:,3)',SOI_main_cycle_hat_tst, 'rmse');
% analysis.mspeSOIm_tst = errperf(ytest(:,3)',SOI_main_cycle_hat_tst, 'mspe');
% analysis.rmspeSOIm_tst = errperf(ytest(:,3)',SOI_main_cycle_hat_tst, 'rmspe');
% analysis.target_range_SOIm_tst = max(ytrain(:,3)) - min(ytrain(:,3));
analysis.nrmseSOIm_tst = ((analysis.rmseSOIm_tst / analysis.target_range_SOIm)) * 100;

analysis.nrmse_sd6_SOIm_tst      = analysis.rmseSOIm_tst   / (6 * analysis.std_SOIm)   * 100;
analysis.nrmse_sd4_SOIm_tst      = analysis.rmseSOIm_tst   / (4 * analysis.std_SOIm)   * 100;
analysis.nrmse_clipped_SOIm_tst  = analysis.rmseSOIm_tst   / analysis.range_clipped_SOIm * 100;
analysis.nmae_SOIm_tst           = analysis.maeSOIm_tst    / analysis.target_range_SOIm  * 100;
analysis.nmae_sd4_SOIm_tst       = analysis.maeSOIm_tst    / (4 * analysis.std_SOIm)     * 100;
analysis.nmae_sd6_SOIm_tst       = analysis.maeSOIm_tst    / (6 * analysis.std_SOIm)     * 100;
analysis.nmae_clipped_SOIm_tst   = analysis.maeSOIm_tst    / analysis.range_clipped_SOIm * 100;


analysis.maeDOIh2_tst = errperf(1e3*ytest(:,4)',1e3*DOI_H2_cycle_hat_tst, 'mae');
analysis.mseDOIh2_tst = errperf(1e3*ytest(:,4)',1e3*DOI_H2_cycle_hat_tst, 'mse');
analysis.rmseDOIh2_tst = errperf(1e3*ytest(:,4)',1e3*DOI_H2_cycle_hat_tst, 'rmse');
analysis.mspeDOIh2_tst = errperf(1e3*ytest(:,4)',1e3*DOI_H2_cycle_hat_tst, 'mspe');
analysis.rmspeDOIh2_tst = errperf(1e3*ytest(:,4)',1e3*DOI_H2_cycle_hat_tst, 'rmspe');
analysis.nrmseDOIh2_tst = ((analysis.rmseDOIh2_tst / analysis.target_range_DOIh2)) * 100;

analysis.nrmse_sd6_DOIh2_tst      = analysis.rmseDOIh2_tst   / (6 * analysis.std_DOIh2)   * 100;
analysis.nrmse_sd4_DOIh2_tst      = analysis.rmseDOIh2_tst   / (4 * analysis.std_DOIh2)   * 100;
analysis.nrmse_clipped_DOIh2_tst  = analysis.rmseDOIh2_tst   / analysis.range_clipped_DOIh2 * 100;
analysis.nmae_DOIh2_tst           = analysis.maeDOIh2_tst    / analysis.target_range_DOIh2  * 100;
analysis.nmae_sd4_DOIh2_tst       = analysis.maeDOIh2_tst    / (4 * analysis.std_DOIh2)     * 100;
analysis.nmae_sd6_DOIh2_tst       = analysis.maeDOIh2_tst    / (6 * analysis.std_DOIh2)     * 100;
analysis.nmae_clipped_DOIh2_tst   = analysis.maeDOIh2_tst    / analysis.range_clipped_DOIh2 * 100;

% rmseDOIm_tst = rmse((ytest(:,1))',(DOI_main_cycle_hat_tst),"all"); % bar
% target_range_DOIm_tst = max(ytest(:,1)) - min(ytest(:,1));
% rmspeDOIm_tst = ((rmseDOIm_tst / target_range_DOIm_tst)) * 100;
% rmseP2M_tst=rmse(ytest(:,2)',P2M_cycle_hat_tst,"all");
% target_range_P2M_tst = max(ytest(:,2)) - min(ytest(:,2));
% rmspeP2M_tst = ((rmseP2M_tst / target_range_P2M_tst)) * 100;
% rmseSOIm_tst=rmse(ytest(:,3)', SOI_main_cycle_hat_tst,"all");
% target_range_SOIm_tst = max(ytest(:,3)) - min(ytest(:,3));
% rmspeSOIm_tst = ((rmseSOIm_tst / target_range_SOIm_tst)) * 100;
% rmseDOIh2_tst=rmse((ytest(:,4))*1e3,DOI_H2_cycle_hat_tst,"all");
% target_range_DOIh2_tst = max((ytest(:,4))*1e3) - min((ytest(:,4))*1e3);
% rmspeDOIh2_tst = ((rmseDOIh2_tst / target_range_DOIh2_tst)) * 100;

SSR_DOIm_tst = sum((ytest(:,1)' - DOI_main_cycle_hat_tst).^2); % Sum of squared residuals
TSS_DOIm_tst = sum(((DOI_main_cycle_hat_tst - mean(DOI_main_cycle_hat_tst)).^2)); % Total sum of squares
analysis.r2DOIm_tst = 1 - SSR_DOIm_tst/TSS_DOIm_tst; % R squared
SSR_P2M_tst = sum((ytest(:,2)' - P2M_cycle_hat_tst).^2); 
TSS_P2M_tst = sum(((P2M_cycle_hat_tst - mean(P2M_cycle_hat_tst)).^2));
analysis.r2P2M_tst = 1 - SSR_P2M_tst/TSS_P2M_tst;
SSR_SOIm_tst = sum((ytest(:,3)' - SOI_main_cycle_hat_tst).^2); 
TSS_SOIm_tst = sum(((SOI_main_cycle_hat_tst - mean(SOI_main_cycle_hat_tst)).^2));
analysis.r2SOIm_tst = 1 - SSR_SOIm_tst/TSS_SOIm_tst;
SSR_DOIh2_tst = sum((ytest(:,4)' - DOI_H2_cycle_hat_tst).^2); 
TSS_DOIh2_tst = sum(((DOI_H2_cycle_hat_tst - mean(DOI_H2_cycle_hat_tst)).^2));
analysis.r2DOIh2_tst = 1 - SSR_DOIh2_tst/TSS_DOIh2_tst;

%% Scatter plot measured vs. predicted (linear ideally) TEST
if plot_pred_test
figure
set(gcf,'color','w');
set(gcf,'units','points','position',[200,200,900,900])

subplot(2,2,1)
scatter(ytest(:,1)*1e3, DOI_main_cycle_hat_tst*1e3, 4, 'k', 'x', 'LineWidth', 5);
max_scale_x = round(max(DOI_main_cycle_hat_tst*1e3), 5);
max_scale_y = round(max(ytest(:,1)*1e3), 5);
min_scale_x = round(min(DOI_main_cycle_hat_tst*1e3), 5);
min_scale_y = round(min(ytest(:,1)*1e3), 5);
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')        
    else
        xlabel({'True DOI Main / ms'},'Interpreter','latex')
        ylabel({'Predicted DOI Main / ms'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.5f ms', analysis.rmseDOIm_tst*1e3); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.5f', analysis.r2DOIm_tst); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,2)
scatter(ytest(:,2), P2M_cycle_hat_tst, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(P2M_cycle_hat_tst), -1);
max_scale_y = round(max(ytest(:,2)), -1);
min_scale_x = round(min(P2M_cycle_hat_tst), -1);
min_scale_y = round(min(ytest(:,2)), -1);
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
else
        xlabel({'True P2M / us'},'Interpreter','latex')
        ylabel({'Predicted P2M / us'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.1f us', analysis.rmseP2M_tst); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.1f', analysis.r2P2M_tst); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,3)
scatter(ytest(:,3), SOI_main_cycle_hat_tst, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(SOI_main_cycle_hat_tst), 1);
max_scale_y = round(max(ytest(:,3)), 1);
min_scale_x = round(min(SOI_main_cycle_hat_tst), 1);
min_scale_y = round(min(ytest(:,3)), 1);
% acc = 0.5; max_scale_x = round(max_scale_x/acc)*acc - 0.5;
% max_scale_y = round(max_scale_y/acc)*acc - 0.5; % MANIP
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        xlabel({'True SOI Main / CADbTDC'},'Interpreter','latex')
        ylabel({'Predicted SOI Main / CADbTDC'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f (bTDC CAD)', analysis.rmseSOIm_tst); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2SOIm_tst); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');


subplot(2,2,4)
scatter(ytest(:,4)*1e3, DOI_H2_cycle_hat_tst*1e3, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(DOI_H2_cycle_hat_tst*1e3), 4);
max_scale_y = round(max(ytest(:,4)*1e3), 4);
min_scale_x = round(min(DOI_H2_cycle_hat_tst*1e3), 4);
min_scale_y = round(min(ytest(:,4)*1e3), 4);
xlim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]); ylim([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)]);
line([min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], [min(min_scale_x, min_scale_y), max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');grid on
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        xlabel({'True DOI H2 / ms'},'Interpreter','latex')
        ylabel({'Predicted DOI H2 / ms'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.YRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.4f ms', analysis.rmseDOIh2_tst*1e3); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.4f', analysis.r2DOIh2_tst); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

if save_plots_sw
    type = "/Prediction_Actual_Test"; 
    resolution = 0;
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, resolution)
end


%% Plotting on test dataset TIME
figure
set(gcf,'color','w');
% set(gcf,'units','points','position',[200,200,900,400]) % paper
set(gcf,'units','points','position',[2030,193,780,600]) % powerpoint 50% 
% set(gcf,'color','w');

max_scale = 4300;

subplot(4,1,1)
plot(ytest(:,1)*1e3', 'r--')
hold on
plot(DOI_main_cycle_hat_tst*1e3,'k-')
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'Main Inj. DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        ylabel('Main Inj. DOI / ms','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
% title('Prediction on Test Data','Interpreter', 'latex')
ax = gca;
% ax.XTick = 0:2000:12000;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
xlim([0,max_scale])
ylim([0.0003*1e3,0.00053*1e3])
legend({'Measured','Predicted'},'Location','northeast','Orientation','horizontal','Interpreter', 'latex')

subplot(4,1,2)
set(gcf,'units','points','position',[200,200,900,400])
plot(ytest(:,2)', 'r--')
hold on
plot(P2M_cycle_hat_tst,'k-')
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ \\ / \mu s')},'Interpreter','latex')
    else
        ylabel({'P2M';'/ us'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        ylabel('P2M / us','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
% ax.XTick = 0:2000:12000;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
xlim([0,max_scale])
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')

subplot(4,1,3)
set(gcf,'units','points','position',[200,200,900,400])
plot(ytest(:,3)', 'r--')
hold on
plot(SOI_main_cycle_hat_tst,'k-')
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ \\ / CADbTDC')},'Interpreter','latex')
    else
        ylabel({'SOI Main';'/ CADbTDC'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        ylabel('SOI Main / CADbTDC','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca; ax.XTickLabel = [];
% ax.XTick = 0:2000:12000;
ax.XRuler.Exponent = 0;
xlim([0,max_scale])
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')

subplot(4,1,4)
set(gcf,'units','points','position',[200,200,900,400])
plot(ytest(:,4)*1e3', 'r--')
hold on
plot(DOI_H2_cycle_hat_tst*1e3,'k-')
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'H2 DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        ylabel('H2 DOI / ms','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
% ax.XTick = 0:2000:12000;
ax.XRuler.Exponent = 0;
xlim([0,max_scale])
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal','Interpreter', 'latex')

if save_plots_sw
    type = "/Prediction_Time_Test";
    resolution = 0;
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, resolution, Opts.multi_lines_ylabel)
end
end

%% Save
Par.nActions = featureDimension;
Par.nOutputs = numResponses; 
Par.TotalLearnables = analysis.TotalLearnables(run_nmbr,1) ;
Par.FinalRMSE = analysis.FinalRMSE(run_nmbr,1);
Par.FinalValidationLoss = analysis.FinalValidationLoss(run_nmbr,1);
Par.ElapsedTime = analysis.ElapsedTime(run_nmbr,1);
Par.Savename = analysis.savename(run_nmbr,1);
% Par.RMSE_Test = [rmseIMEP_tst, rmseNOx_tst, rmseSOOT_tst, rmseMPRR_tst];
% Par.RMSE_Val = [rmseIMEP_val, rmseNOx_val, rmseSOOT_val, rmseMPRR_val];
% Par.RMSPE_Test = [rmspeIMEP_tst, rmspeNOx_tst, rmspeSOOT_tst, rmspeMPRR_tst];
% Par.RMSPE_Val = [rmspeIMEP_val, rmspeNOx_val, rmspeSOOT_val, rmspeMPRR_val];

if save_analysis == true
    save(['../Results/Analysis_',savename],"analysis")
end

if do_training == true
    save(['../Results/Par_',savename],"Par")
end

trainingrun = trainingrun + 1; % increase when doing grid search

if break_loop break
end

end

end

%% EXTERNAL FUNCTIONS
function save_plots(gcf, MP, trainingrun, type, zoom, resolution, multi_lines_ylabel, externalize, yyxais_right)
    if (~exist('resolution', 'var'))
        resolution = 150;
    end
    if (~exist('externalize', 'var'))
        externalize = false;
    end
    if (~exist('zoom', 'var'))
        zoom = false;
    end
    if (~exist('yyxais_right', 'var'))
        yyxais_right = false;
    end
    if (~exist('multi_lines_ylabel', 'var'))
        multi_lines_ylabel = false;
    end

    if zoom
        figFileName="../Plots/"+ sprintf("%04d",MP)+'/'+ sprintf('%04d',trainingrun) + type + '_zoom';
    else
        figFileName="../Plots/"+ sprintf("%04d",MP)+'/'+ sprintf('%04d',trainingrun) + type;
    end
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    % saveas(gcf,figFileName,"epsc");
    % saveas(gcf,figFileName,"pdf");
    if resolution > 0
        cleanfigure('targetResolution', resolution)
    end
    if multi_lines_ylabel
        axis_code = 'ylabel style={align=center}, scaled ticks=false,';
    else
        axis_code = 'scaled ticks=false, ';
    end
    if yyxais_right
        axis_code = 'ylabel style={align=center}, axis y line*=right, every outer x axis line/.append style={draw=none},every x tick label/.append style={text opacity=0},every x tick/.append style={draw=none},';
    end
    if externalize
        matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions',axis_code, 'externalData',true);
    else
        matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions', axis_code);
    end
    % axis_code = 'yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, scaled ticks=false,';
    % axis_code = 'yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, scaled ticks=false,';
    % matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions',axis_code); % 'externalData',true
    
    %export_fig(figFileName,'-eps');

end
