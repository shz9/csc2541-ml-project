% Test GPML covSM on airline dataset
% Andrew Gordon Wilson, 11 Oct 2013

clear all

load CO2data

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

[mSM s2] = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y, z);

figure(1); clf;
hold on;

f = [mSM + 2*sqrt(s2); flipdim(mSM - 2*sqrt(s2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

plot(xtrain,ytrain,'b','LineWidth',2);

plot(xtest,ytest,'g','LineWidth',2);


likfunc = @likGauss;
covfunc = @covSEiso; 
hypSE.cov = log([40 std(y)]); hypSE.lik = log(sn);
hypSEtrain = minimize(hypSE, @gp, -100, @infExact, [], covfunc, likfunc, x,y);
[mSE s2SE] = gp(hypSEtrain, @infExact, [], covfunc, likfunc, x,y,z);

figure(1); %plot(xtest,mSE,'r','LineWidth',2);

plot(xtest,mSM,'k','LineWidth',2);

xlabel('Year');
ylabel('TSI');
