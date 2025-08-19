function [utrain, ytrain, uval, yval, utest, ytest] = getDatasets_IL(data_conc, idx_start, idx_end, ratio_train, ratio_val, plot_init, kill_violated_data, kill_points_zero_h2_doi, RNG)

% data
% engine_deg = data_conc.fpga_lastCycle_engine_angle(idx_start:end); % last cycle? check necessary
if idx_end == 0
    idx_end = length(data_conc.fpga_lastCycle_engine_speed);
end

rpm = data_conc.fpga_lastCycle_engine_speed(idx_start:idx_end); % constant at 1500 rpm

RNG_Active = data_conc.mabx_Control_Mode(idx_start:idx_end); % =2 when random generator is active
Limit_hit_lvl1 = data_conc.mabx_RngStop_Level1(idx_start:idx_end); % =1 when limits have been hit - Level 1
Limit_hit_lvl2 = data_conc.mabx_RngStop_Level2(idx_start:idx_end); % =1 when limits have been hit - Level 2
% plot(Limit_hit_lvl1); hold on; plot(Limit_hit_lvl2); hold off;

if plot_init
    % plot Limits and RNG
    % figure;
    % plot (Limit_hit_lvl1);
    % hold on;
    % plot(Limit_hit_lvl2);
    % plot(RNG_Active);
    % legend("Limit Lvl1","Limit Lvl 2","Controller Mode (RNG = 2)");
    % hold off;
    
    % counter for hits
    cntr_lvl1 = 0;
    for i=1:length(Limit_hit_lvl1)
        if Limit_hit_lvl1(i) == 1 
           cntr_lvl1 = cntr_lvl1 + 1;
        end
    end
    disp(cntr_lvl1)

    % counter for hits
    cntr_lvl2 = 0;
    for i=1:length(Limit_hit_lvl2)
        if Limit_hit_lvl2(i) == 1 
           cntr_lvl2 = cntr_lvl2 + 1;
        end
    end
    disp(cntr_lvl2)
end


DOI_main = data_conc.mabx_DOImain(idx_start:idx_end);
DOI_pre = data_conc.mabx_DOIpre(idx_start:idx_end);
SOI_pre = data_conc.mabx_SOIpre(idx_start:idx_end); % usual soi 
SOI_main = data_conc.mabx_SOImain(idx_start:idx_end);
H2_doi = data_conc.mabx_DOI_H2(idx_start:idx_end); % convert s to ms
NOx = data_conc.can_NOx_NOx(idx_start:idx_end); % cheap CAN sensor, not FTIR! in ppm
Soot = data_conc.mabx_SOOT(idx_start:idx_end); % in mgm3
IMEP = data_conc.fpga_lastCycle_IMEP(idx_start:idx_end); % pressure, bascically load in MPa
MPRR = data_conc.fpga_lastCycle_MPRR(idx_start:idx_end); % dp max, in Mpa per CAD
% Imep ref
IMEP_ref0 = data_conc.C2C_NMPC_IMEP_load_ref0(idx_start:idx_end)*1e5;
IMEP_ref1 = data_conc.C2C_NMPC_IMEP_load_ref1(idx_start:idx_end)*1e5;
IMEP_ref2 = data_conc.C2C_NMPC_IMEP_load_ref2(idx_start:idx_end)*1e5;

% calculate P2M
% P2M = data_conc.mabx_P2M; % P2M time in micro seconds
P2M = (SOI_pre-SOI_main)./360;
P2M = P2M./(rpm./60);
P2M = (P2M-DOI_pre).*1e6;

%% Remove cycles when random input is not activated
% H2_doi_cycle = H2_doi_cycle.*(Limit_hit_lvl1==0);% why is that here?

% index_ACT = find(RNG_Active==2 || Limit_hit_lvl1 ==0 || Limit_hit_lvl1 ==0); % check for
% both lvl1 and RNG off
if RNG == true
    index_ACT = find(RNG_Active == 2); %
else
    index_ACT = find(RNG_Active == 3); %
    idx_end = length(index_ACT);
end    

if kill_violated_data
    DOI_main_cycle = DOI_main(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    DOI_main_cycle(DOI_main_cycle == 0) = [];   

    SOI_main_cycle = SOI_main(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    SOI_main_cycle(SOI_main_cycle == 0) = []; 
    
    P2M_cycle = P2M(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    P2M_cycle(P2M_cycle == 0) = []; 
    
    for i = 1:length(H2_doi)
        if H2_doi(i) == 0
               H2_doi(i) = 123456; % invalid value, quick fix, because soot gets to 0 quite often
        end
    end
    H2_doi_cycle = H2_doi(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    H2_doi_cycle(H2_doi_cycle == 0) = []; 
    for i = 1:length(H2_doi_cycle)
        if H2_doi_cycle(i) == 123456
               H2_doi_cycle(i) = 0; % invalid value, quick fix, because soot gets to 0 quite often
        end
    end
  
    P2M_cycle = P2M(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    P2M_cycle(P2M_cycle == 0) = []; 
  
    NOx_cycle = NOx(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    NOx_cycle(NOx_cycle == 0) = []; 
    
    for i = 1:length(Soot)
        if Soot(i) == 0
               Soot(i) = 123456; % invalid value, quick fix, because soot gets to 0 quite often
        end
    end
    Soot_cycle = Soot(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    Soot_cycle(Soot_cycle == 0) = [];
    for i = 1:length(Soot_cycle)
        if Soot_cycle(i) == 123456
               Soot_cycle(i) = 0; % invalid value, quick fix, because soot gets to 0 quite often
        end
    end
  
    IMEP_cycle = IMEP(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    IMEP_cycle(IMEP_cycle == 0) = []; 
    
    MPRR_cycle = MPRR(index_ACT).*(Limit_hit_lvl1(index_ACT) == 0).*(Limit_hit_lvl2(index_ACT) == 0); % cycle naming means valid
    MPRR_cycle(MPRR_cycle == 0) = []; 

else
    DOI_main_cycle = DOI_main(index_ACT);
    % DOI_pre_cycle = DOI_pre(index_ACT);
    % SOI_pre_cycle = SOI_pre(index_ACT);
    SOI_main_cycle = SOI_main(index_ACT);
    P2M_cycle = P2M(index_ACT);
    NOx_cycle = NOx(index_ACT);
    Soot_cycle = Soot(index_ACT);
    IMEP_cycle = IMEP(index_ACT);
    MPRR_cycle = MPRR(index_ACT);
    IMEP_ref0_cycle = IMEP_ref0(index_ACT);
    IMEP_ref1_cycle = IMEP_ref1(index_ACT);
    IMEP_ref2_cycle = IMEP_ref2(index_ACT);    
    H2_doi_cycle = H2_doi(index_ACT);

    if kill_points_zero_h2_doi

        for i = 1:length(Soot_cycle)
            if Soot_cycle(i) == 0
                   Soot_cycle(i) = 123456; % invalid value, quick fix, because soot gets to 0 quite often
            end
        end
        
        for i = 1:length(H2_doi_cycle)
            if H2_doi_cycle(i) == 0
               H2_doi_cycle(i) = 0;
               DOI_main_cycle(i) = 0;
               SOI_main_cycle(i) = 0;
               P2M_cycle(i) = 0;
               NOx_cycle(i) = 0;
               Soot_cycle(i) = 0;
               IMEP_cycle(i) = 0;
               MPRR_cycle(i) = 0;
           end
        end
        H2_doi_cycle(H2_doi_cycle == 0) = []; 
        DOI_main_cycle(DOI_main_cycle == 0) = []; 
        DOI_main_cycle(DOI_main_cycle == 0) = []; 
        SOI_main_cycle(SOI_main_cycle == 0) = []; 
        P2M_cycle(P2M_cycle == 0) = []; 
        NOx_cycle(NOx_cycle == 0) = []; 
        IMEP_cycle(IMEP_cycle == 0) = []; 
        MPRR_cycle(MPRR_cycle == 0) = []; 
        
        Soot_cycle(Soot_cycle == 0) = []; 
        for i = 1:length(Soot_cycle)
            if Soot_cycle(i) == 123456
                   Soot_cycle(i) = 0; % invalid value, quick fix, because soot gets to 0 quite often
            end
        end
    end
end

%% Create previous cycle inputs
IMEP_cycle = IMEP_cycle(2:idx_end);
IMEP_old = IMEP_cycle(1);
IMEP_old(2:length(IMEP_cycle)) = IMEP_cycle(1:end-1);

% DOI_pre_cycle = DOI_pre_cycle(1:end-1);
DOI_main_cycle = DOI_main_cycle(1:end-1);
P2M_cycle = P2M_cycle(1:end-1);
% SOI_pre_cycle = SOI_pre_cycle(1:idx_end-1);
SOI_main_cycle = SOI_main_cycle(1:end-1);
H2_doi_cycle = H2_doi_cycle(1:end-1);
NOx_cycle = NOx_cycle(1:end-1);
Soot_cycle = Soot_cycle(1:end-1);
MPRR_cycle = MPRR_cycle(2:end);
IMEP_ref0_cycle = IMEP_ref0_cycle(1:end-1);
IMEP_ref1_cycle = IMEP_ref1_cycle(1:end-1);
IMEP_ref2_cycle = IMEP_ref2_cycle(1:end-1);

%% Datasplit Definiton
dataset_length = size(IMEP_cycle, 2);
indx_tr = floor(ratio_train*dataset_length);
indx_tst = floor(ratio_val*dataset_length);

% assign dataset variables IL!
y1 = DOI_main_cycle';
y2 = P2M_cycle';
y3 = SOI_main_cycle';
y4 = H2_doi_cycle';

u1 = IMEP_ref0_cycle';
u2 = IMEP_ref1_cycle';
u3 = IMEP_ref2_cycle';
u4 = IMEP_old'; % feedback of old IMEP

utrain = [u1(1:indx_tr)'; u2(1:indx_tr)'; u3(1:indx_tr)'; u4(1:indx_tr)'];
ytrain = [y1(1:indx_tr)'; y2(1:indx_tr)'; y3(1:indx_tr)'; y4(1:indx_tr)'];

uval = [u1(indx_tr:indx_tst)'; u2(indx_tr:indx_tst)'; u3(indx_tr:indx_tst)'; u4(indx_tr:indx_tst)'];
yval = [y1(indx_tr:indx_tst)'; y2(indx_tr:indx_tst)'; y3(indx_tr:indx_tst)'; y4(indx_tr:indx_tst)'];

utest = [u1(indx_tst:end)'; u2(indx_tst:end)'; u3(indx_tst:end)'; u4(indx_tst:end)'];
ytest = [y1(indx_tst:end)'; y2(indx_tst:end)'; y3(indx_tst:end)'; y4(indx_tst:end)'];
