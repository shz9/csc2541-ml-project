function [sm_spect w mus sigmas] = smspect(s,hyp,Q)

w = exp(hyp(1:Q));
mus = exp(hyp(Q+1:2*Q));
sigmas = exp(hyp(2*Q+1:3*Q));

sm_spect = zeros(numel(s),1);

for j=1:Q
    sm_spect = sm_spect + w(j)*(normpdf(s,mus(j),sigmas(j)) + normpdf(-s,mus(j),sigmas(j)));
end
    