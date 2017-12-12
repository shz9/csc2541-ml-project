
% Test empirical spectrum method

load airline

N = numel(ytrain);
emp_spect = abs(fft(ytrain)).^2/N;
log_emp_spect = log(emp_spect);

M = floor(N/2);

s = [[0:M],[-M+1:1:-1]]'/N; 

s = s(1:M+1);
emp_spect = emp_spect(1:M+1);
log_emp_spect = log_emp_spect(1:M+1);


figure(2); clf;
plot(s,log_emp_spect)

figure(3)
plot(s,emp_spect);

% careful, you are using the log_emp_spect here

% the_density = emp_spect;
the_density = log_emp_spect;

numcdf = @(j) trapz(j,the_density);

total_area = trapz(s,the_density);

spec_cdf = zeros(numel(s,1),1);

spec_cdf(1)=0;

for i=2:numel(s)
    spec_cdf(i) = trapz(s(1:i),the_density(1:i));
end

spec_cdf = spec_cdf ./total_area;

figure(4);
plot(s,spec_cdf)

nsamp = 1e4;

a = rand(nsamp,1);

invsamps = zeros(numel(a),1);

tic
for i=1:numel(a);
  invsamps(i) = inv_spec_cdf(a(i),s,spec_cdf);
end
toc

figure(5)
hist(invsamps,100)


options = statset('Display','final','MaxIter',100);
obj = gmdistribution.fit(invsamps,10,'Options',options);

mus = obj.mu;
ws = obj.PComponents;


obj.mu
sqrt(obj.Sigma)

