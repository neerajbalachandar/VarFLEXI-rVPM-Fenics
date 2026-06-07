clear; clc; close all;

%% =====================================================
%% LOAD EXPERIMENTAL DATA
%% =====================================================

filepath = '/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/expt_data/bifurcation_test_K_0_slpm_20.mat';

S = load(filepath);

TT = S.acquired_data;

%% =====================================================
%% EXTRACT SIGNAL
%% =====================================================

vars = TT.Properties.VariableNames;

signal = TT.(vars{1});

t = seconds(TT.Time - TT.Time(1));

%% =====================================================
%% BASIC INFORMATION
%% =====================================================

dt = mean(diff(t));
fs = 1/dt;

signal_mean = mean(signal);
signal_std  = std(signal);

fprintf('\n=========================================\n');
fprintf('EXPERIMENTAL DATA INFORMATION\n');
fprintf('=========================================\n\n');

fprintf('Channel name      : %s\n', vars{1});
fprintf('Sampling frequency: %.2f Hz\n', fs);
fprintf('Time duration     : %.2f s\n', t(end));
fprintf('Total samples     : %d\n', length(signal));

fprintf('\nSignal statistics:\n');

fprintf('Mean value        : %.10e\n', signal_mean);
fprintf('Std deviation     : %.10e\n', signal_std);
fprintf('Minimum value     : %.10e\n', min(signal));
fprintf('Maximum value     : %.10e\n', max(signal));

fprintf('\nLikely interpretation:\n');

fprintf(['- Signal is probably raw DAQ voltage or ', ...
         'strain-gauge bridge output.\n']);

fprintf(['- Units are NOT stored explicitly in the MAT file.\n']);

fprintf(['- Magnitude (~1e-4) suggests volts or normalized ', ...
         'bridge voltage.\n']);

fprintf(['- Physical displacement is NOT directly stored.\n']);

fprintf('\n=========================================\n\n');

%% =====================================================
%% RAW SIGNAL PLOT
%% =====================================================

figure;

plot(t, signal, 'LineWidth',1);

xlabel('Time (s)');
ylabel('Raw Signal');

title('Experimental Raw Signal');

grid on;

%% =====================================================
%% BETTER VISUALIZATION
%% =====================================================

% Remove DC offset
signal_fluct = signal - mean(signal);

figure;

plot(t, signal_fluct, 'LineWidth',1);

xlabel('Time (s)');
ylabel('Fluctuation About Mean');

title('Mean-Removed Signal');

grid on;

%% =====================================================
%% ZOOMED VIEW
%% =====================================================

Nzoom = 4000;

figure;

plot(t(1:Nzoom), signal_fluct(1:Nzoom), ...
     'LineWidth',1);

xlabel('Time (s)');
ylabel('Fluctuation');

title('Zoomed Oscillation View');

grid on;