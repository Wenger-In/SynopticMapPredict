clear; close all;
CR = 2097;
file_name = ['cr',num2str(CR)];
GONG_dir = 'E:\Research\Data\GONG\fits\';
WSO_dir = 'E:\Research\Data\WSO\field\';
GONG_recon_dir = 'E:\Research\Work\magnetic_multipole\harmonics_map\GONG\';
WSO_recon_dir = 'E:\Research\Work\magnetic_multipole\harmonics_map\WSO\';

GONG_map = importdata([GONG_dir,file_name,'.mat']); % [Gs]
WSO_map = importdata([WSO_dir,file_name,'.mat']); % [nT]
GONG_recon_map = importdata([GONG_recon_dir,file_name,'.mat']); % [Gs]
WSO_recon_map = importdata([WSO_recon_dir,file_name,'.mat']); % [nT]

figure();
histogram(GONG_map,'Normalization','probability','EdgeColor','none');
hold on
histogram(WSO_map./100,'Normalization','probability','FaceColor','none','EdgeColor','r','LineWidth',2);
legend('GONG map','WSO map')
xlim([-5 5])
set(gca,'LineWidth',2,'FontSize',20)
title(['CR',num2str(CR)])

figure();
histogram(GONG_recon_map,'Normalization','probability','EdgeColor','none');
hold on
histogram(WSO_recon_map./100,'Normalization','probability','FaceColor','none','EdgeColor','r','LineWidth',2);
legend('GONG reconstructed map','WSO reconstructed map')
xlim([-5 5])
set(gca,'LineWidth',2,'FontSize',20)
title(['CR',num2str(CR)])
