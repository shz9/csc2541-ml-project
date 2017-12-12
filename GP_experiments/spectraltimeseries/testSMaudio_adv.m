% Test GPML covSM on audio dataset
% Andrew Gordon Wilson, March 2014

clear all

load audio1

x = xtrain;
y = ytrain;

z = xtest;

s = 42; randn('seed',s)

D = 1;  % Input dimensionality
Q = 7;  % Number of covSM components

numinit = 10;   
  
%smhyp_try = initSMhypers(Q,x,y);
% try with just the first set
smhyp_try = initSMhypersadvanced(Q,x,y,1);
%smhyp_try = initSMhypersadvanced(Q,xtemp1,ytemp1,1);    % initialise SM hypers


likfunc = @likGauss; sn = 0.1;
covfunc = {@covSM,Q}; hypspec.cov = smhyp_try; hypspec.lik = log(sn);

winc = zeros(size(hypspec.cov));
ninc = zeros(1,1);

fs = []
for epoch = 1:1e3
    epoch
    
  order = randperm(size(x,1));
  [f dw] = gp(hypspec,@infExact, [], covfunc, likfunc, x(order(1:50)),y(order(1:50)));
  dn = dw.lik;
  dw = dw.cov;
  winc = 0.95*winc + 1e-8*dw;  % advice on tuning these parameters?
  ninc = 0.95*ninc + 1e-8*dn;
  
  hypspec.cov = hypspec.cov - winc;
  hypspec.lik = hypspec.lik - ninc;
  
  fs(end+1) = f;
end


smhyp_train = minimize(hypspec, @gp, -200, @infExact, [], covfunc, likfunc, x, y);
nlml_final = gp(smhyp_train, @infExact, [], covfunc, likfunc, x, y)


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

figure(1); plot(xtest,mSE,'r','LineWidth',2);

plot(xtest,mSM,'k','LineWidth',2);

xlabel('Months');
ylabel('CO_2 (ppm)');
