% Test GPML covSM on airline dataset

clear all

load airlinedata

x = xtrain;
y = ytrain;

z = xtest;

s = 42; randn('seed',s)

D = 1;  % Input dimensionality
Q = 10; % Number of covSM components

numinit = 10;

nlml = Inf;

% try numinit random initialisations
for j=1:numinit

smhyp_try = initSMhypers(Q,x,y);    % initialise SM hypers

% Use spectral mixture to do regression

likfunc = @likGauss; sn = 1;
covfunc = {@covSM,Q}; hypspec.cov = smhyp_try; hypspec.lik = log(sn);


% short optimization run for each of these numinit initialisations
% you may want to change shortrun to simply -1, and subsequently 
% increase numinit.  play with the tradeoff between numinit and 
% shortrun to see how it affects the (consistency of the) results.

shortrun = 100;

smhyp_train = minimize(hypspec, @gp, -shortrun, @infExact, [], covfunc, likfunc, x, y);
nlml_new = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y);

if (nlml_new < nlml)
    smhyp_init = smhyp_try;
    nlml = nlml_new;
end

end

likfunc = @likGauss; sn = 1;
covfunc = {@covSM,Q}; hypspec.cov = smhyp_init; hypspec.lik = log(sn);

smhyp_train = minimize(hypspec, @gp, -1000, @infExact, [], covfunc, likfunc, x, y);
nlml = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y);

% Final nlml should be about 350

[mSM s2] = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y, z);

figure(1); clf;
hold on;

f = [mSM + 2*sqrt(s2); flipdim(mSM - 2*sqrt(s2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);

plot(xtrain,ytrain,'b','LineWidth',2);

plot(xtest,ytest,'g','LineWidth',2);

plot(xtest,mSM,'k','LineWidth',2);
