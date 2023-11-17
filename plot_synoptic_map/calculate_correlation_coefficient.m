clear; close all;
cr = 2259;
%% import data
obs_dir = 'E:\Research\Work\magnetic_multipole\harmonics_map\WSO\';
obs_file = [obs_dir,'cr',num2str(cr),'.mat'];
obs_data = load(obs_file);
magneto_obs = obs_data.magneto;

pred_dir = 'E:\Research\Work\magnetic_multipole\predict\';
pred_file = [pred_dir,'cr',num2str(cr),'_pred.mat'];
pred_data = load(pred_file);
magneto_pred = pred_data.magneto;
%% interpolate into rougher grid
raw_lon = linspace(0,360,360);
raw_lat = linspace(-90,90,180);
[raw_llon,raw_llat] = meshgrid(raw_lon,raw_lat);
std_lon = linspace(0,360,73);
std_lat = linspace(-90,90,37);
[std_llon,std_llat] = meshgrid(std_lon,std_lat);

Br_obs = interp2(raw_llon,raw_llat,magneto_obs,std_llon,std_llat);
Br_pred = interp2(raw_llon,raw_llat,magneto_pred,std_llon,std_llat);
%% calculate correlation coefficient
Br_obs = reshape(Br_obs,1,[]);
Br_pred = reshape(Br_pred,1,[]);
cc = corrcoef(Br_obs,Br_pred);
p = polyfit(Br_obs,Br_pred,1);
xFit = linspace(min(Br_obs), max(Br_obs), 100);
yFit = polyval(p, xFit);
%% plot figure
figure();
scatter(Br_obs,Br_pred,3);
hold on;
plot(xFit,yFit,'r','LineWidth',2);
grid on
axis equal
xlim([-4,4])
ylim([-4,4])
xlabel('Observation [G]')
ylabel('Prediction [G]')
title(['CR',num2str(cr),',k=',num2str(p(1)),',b=',num2str(p(2)),',CC=',num2str(cc(1,2))]);
set(gca,'LineWidth',2,'FontSize',15)