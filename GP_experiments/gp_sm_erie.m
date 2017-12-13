% All files from 'Spectral Mixture Kernels' (https://people.orie.cornell.edu/andrew/code/) should be downloaded
% load data
lake = readtable("lake_erie.csv");
xtrain = table2array(lake(1:round(0.7*size(lake,1)), 1)); ytrain = table2array(lake(1:round(0.7*size(lake,1)), 3));
xtest = table2array(lake(round(0.7*size(lake,1))+1:end, 1)); ytest = table2array(lake(round(0.7*size(lake,1))+1:end, 3));

x = xtrain;
y = ytrain;

z = xtest;

s = 8; randn('seed',s)

D = 1;  % Input dimensionality
Q = 4;  % Number of covSM components

numinit = 100;   

nlml = Inf;  

% try numinit random initialisations
for j=1:numinit

smhyp_try = initSMhypers(Q,x,y);    % initialise SM hypers

% Use spectral mixture to do regression

likfunc = @likGauss; sn = 0.1;
covfunc = {@covSM,Q}; hypspec.cov = smhyp_try; hypspec.lik = log(sn);

smhyp_train = minimize(hypspec, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
nlml_new = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y);

if (nlml_new < nlml)
    smhyp_init = smhyp_try;
    nlml = nlml_new;
end

end

likfunc = @likGauss; sn = 0.1;
covfunc = {@covSM,Q}; hypspec.cov = smhyp_init; hypspec.lik = log(sn);

smhyp_train = minimize(hypspec, @gp, -1000, @infExact, [], covfunc, likfunc, x, y);
nlml = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y);

% Final nlml should be about 65
[mSMtrain s2train] = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y, xtrain);
[mSM s2] = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y, z);
% figure
figure(1); clf;
hold on;
f = [mSM + 2*sqrt(s2); flipdim(mSM - 2*sqrt(s2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
f = [mSMtrain + 2*sqrt(s2train); flipdim(mSMtrain - 2*sqrt(s2train),1)];
fill([xtrain; flipdim(xtrain,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

actual = plot([xtrain' xtest'],[ytrain' ytest'],'b','LineWidth',1);
figure(1);
plot(xtrain,mSMtrain,'r','LineWidth',1);
pred = plot(xtest,mSM,'r','LineWidth',1);

xlim([min(xtrain) max(xtest)]); ylim([5 30]);
line([xtrain(round(0.7*size(lake,1))) xtrain(round(0.7*size(lake,1)))], ylim, 'LineStyle', '--', 'Color', 'black', 'LineWidth', 2)
xlabel('Months');
ylabel('Water Level (m)');
print(gcf,'gp_sm_erie.png','-dpng','-r300'); 
