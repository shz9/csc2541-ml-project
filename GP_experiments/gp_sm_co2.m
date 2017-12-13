% All files from 'Spectral Mixture Kernels' (https://people.orie.cornell.edu/andrew/code/) should be downloaded 
% load data
co2 = readtable("mauna-loa-atmospheric-co2.csv");
xtrain = table2array(co2(1:round(0.7*size(co2,1)), 2)); ytrain = table2array(co2(1:round(0.7*size(co2,1)), 1));
xtest = table2array(co2(round(0.7*size(co2,1))+1:end, 2)); ytest = table2array(co2(round(0.7*size(co2,1))+1:end, 1));

x = xtrain;
y = ytrain;

z = xtest;

s = 42; randn('seed',s)

D = 1;  % Input dimensionality
Q = 4;  % Number of covSM components

numinit = 10;   

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
fill([(round(0.7*size(co2,1))+1:size(co2,1))'; flipdim((round(0.7*size(co2,1))+1:size(co2,1))',1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
f = [mSMtrain + 2*sqrt(s2train); flipdim(mSMtrain - 2*sqrt(s2train),1)];
fill([(1:round(0.7*size(co2,1)))'; flipdim((1:round(0.7*size(co2,1)))',1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

actual = plot(1:size(co2,1),[ytrain' ytest'],'b','LineWidth',1);
figure(1);
plot(1:round(0.7*size(co2,1)),mSMtrain,'r','LineWidth',1);
pred = plot(round(0.7*size(co2,1))+1:size(co2,1),mSM,'r','LineWidth',1);

xlim([1 size(co2,1)]); 
line([round(0.7*size(co2,1)) round(0.7*size(co2,1))], ylim, 'LineStyle', '--', 'Color', 'black', 'LineWidth', 2)
xlabel('Months');
ylabel('CO2 Concentration (PPM)');
print(gcf,'gp_sm_co2.pdf','-dpng','-r300'); 
