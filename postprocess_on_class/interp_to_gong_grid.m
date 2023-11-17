clear; close all;
% predicted field
pred_dir = 'E:\Research\Work\magnetic_multipole\harmonics_map\WSO\';
pred_name = 'cr2259.mat';
pred_Br = load([pred_dir,pred_name]); % [uT]
pred_Br = pred_Br.magneto / 100; % [G]
pred_Br = flipud(pred_Br);
% predicted grid
lon_pred = linspace(0,360,360);
lat_pred = linspace(-90,90,180);
[llon_pred,llat_pred] = meshgrid(lon_pred,lat_pred);
% GONG grid
lon_gong = linspace(0,360,360);
lat_sin_gong = linspace(-1,1,180);
lat_gong = asind(lat_sin_gong);
[llon_gong,llat_gong] = meshgrid(lon_gong,lat_gong);
% interpolate predicted field into gong grid
pred_Br_interp = interp2(llon_pred, llat_pred, pred_Br, llon_gong, llat_gong, 'linear');