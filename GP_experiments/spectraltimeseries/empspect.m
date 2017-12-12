function [emp lemp s] = empspect(y)

N = numel(y);
emp = abs(fft(y)).^2/N;
lemp = log(emp);

M = floor(N/2);

s = [[0:M],[-M+1:1:-1]]'/N; 

s = s(1:M+1);
emp = emp(1:M+1);
lemp = lemp(1:M+1);
