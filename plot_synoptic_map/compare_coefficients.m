clear; close all;
obs_path = 'E:\Research\Data\WSO\harmonics\';
pred_path = 'E:\Research\Work\magnetic_multipole\predict\harmonics\';
cr = 2259;

obs_data = load([obs_path, 'cr', num2str(cr), '.dat']);
pred_data = load([pred_path, 'cr', num2str(cr), '.dat']);
pers_data = load([obs_path, 'cr2258.dat']);

obs_coef = obs_data(1:21,3:4);
obs_coef = reshape(obs_coef,1,[]);
pred_coef = pred_data(:,3:4);
pred_coef = reshape(pred_coef,1,[]);
pers_coef = pers_data(1:21,3:4);
pers_coef = reshape(pers_coef,1,[]);

figure()
plot(obs_coef,'k')
hold on
plot(pred_coef,'r')
hold on
plot(pers_coef,'b')
xlabel('coefficient index')
ylabel('coefficient value')
title('harmonic coefficients comparison')
legend('observation','prediction','persistent')

figure()
plot((pred_coef-obs_coef)./obs_coef, 'r')
hold on
plot((pers_coef-obs_coef)./obs_coef, 'b')
hold on
yline(0)
xlabel('coefficient index')
ylabel('coefficient value')
title('harmonic coefficients error = (model output - obs)/obs')
legend('prediction','persistent')