% Initialisation script for SM kernel using the empirical spectral density.  
% Andrew Gordon Wilson, 8 Mar 2014

% This is not an all purpose initialisation script.  Common sense is 
% still required when initialising the SM kernel in new situations. 

% Assumes D=1

function hypinit = initSMhypersadvanced(Q,x,y,flag)

[N,D] = size(x);

% create hypers
w = zeros(1,Q);
m = zeros(D,Q);
s = zeros(D,Q);

% create initialisation vector of all hypers
hypinit = zeros(Q+2*D*Q,1);

emp_spect = abs(fft(y)).^2/N;
log_emp_spect = abs(log(emp_spect));

M = floor(N/2);

freq = [[0:M],[-M+1:1:-1]]'/N; 

freq = freq(1:M+1);
emp_spect = emp_spect(1:M+1);
log_emp_spect = log_emp_spect(1:M+1);


figure(2); clf;
plot(freq,log_emp_spect)

figure(3)
plot(freq,emp_spect);

% careful, you are using the log_emp_spect here

if (flag == 1)
    the_density = emp_spect;
else
    the_density = log_emp_spect;
end

total_area = trapz(freq,the_density);

%spec_cdf = zeros(numel(freq),1);

%spec_cdf(1)=0;

%for i=2:numel(freq)
%    spec_cdf(i) = trapz(freq(1:i),the_density(1:i));
%end

spec_cdf = cumtrapz(freq,the_density);

spec_cdf = spec_cdf ./total_area;

figure(4);
plot(freq,spec_cdf)

nsamp = 1e4;

a = rand(nsamp,1);

invsamps = zeros(numel(a),1);

tic
for i=1:numel(a);
  invsamps(i) = inv_spec_cdf(a(i),freq,spec_cdf);
end
toc

figure(5)
hist(invsamps,1000)


options = statset('Display','final','MaxIter',1000);
obj = gmdistribution.fit(invsamps,Q,'Options',options,'Replicates',20);

m(1,:) = obj.mu;
s(1,:) = sqrt(reshape(obj.Sigma,1,Q));%/N%(x(end)-x(1));    % Check this line!
w(1,:) = std(y).*obj.PComponents;

 test = pdf(obj, [freq]);
 
figure(3); plot(freq,emp_spect); hold on; plot(freq,test*total_area,'m');

hypinit(1:Q) = log(w);   
hypinit(Q+(1:Q*D)) = log(m(:));
hypinit(Q+Q*D+(1:Q*D)) = log(s(:));

% now correct the weights and length-scales




