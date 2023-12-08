clear; close all;

data_dir = 'E:\Research\Work\magnetic_multipole\sunspot\';
train = importdata([data_dir,'train_predict.csv']);
val = importdata([data_dir,'val_predict.csv']);
test = importdata([data_dir,'test_predict.csv']);
future = importdata([data_dir,'future_predict.csv']);
future_llim = importdata([data_dir,'future_predict_llim.csv']);
future_ulim = importdata([data_dir,'future_predict_ulim.csv']);

sn_dir = 'E:\Research\Data\Sunspot\';
sn_info = importdata([sn_dir,'SN_m_tot_V2.0.csv']);
sn = sn_info(:,4);

look_back = 359;
future_step = 120;
index = 1:length(sn) + future_step;
epoch = 1749.042 + (index - 1)/12;

figure();
LineWidth = 2;
FontSize = 15;

plot(epoch(1:length(sn)),sn,'k','LineWidth',LineWidth); hold on
plot(epoch(look_back+1:length(train)+look_back),train,'b','LineWidth',LineWidth); hold on
plot(epoch(look_back+length(train)+1:length(val)+length(train)+look_back),val,'Color','#D95319','LineWidth',LineWidth); hold on
plot(epoch(look_back+length(train)+length(val)+1:length(test)+length(train)+length(val)+look_back),test,'g','LineWidth',LineWidth); hold on
plot(epoch(length(sn)+1:length(sn)+future_step),future,'r','LineWidth',LineWidth); hold on
plot(epoch(length(sn)+1:length(sn)+future_step),future_ulim,':','Color','#7E2F8E','LineWidth',LineWidth); hold on
plot(epoch(length(sn)+1:length(sn)+future_step),future_llim,':','Color','#A2142F','LineWidth',LineWidth); grid on
legend('ISN','train prediction','val prediction','test prediction','future prediction','future prediction upper limit','future prediction lower limit')
xlim([epoch(1),epoch(end)+5])
ylim([0,inf])
xlabel('Year')
ylabel('Sunspot Number')
set(gca,'LineWidth',LineWidth,'FontSize',FontSize)