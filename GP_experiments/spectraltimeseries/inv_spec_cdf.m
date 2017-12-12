% inverse cdf for empirical spectral density
% Andrew Gordon Wilson, March 2014

function invspec = inv_spec_cdf(val,s,spec_cdf)

[ignore bin] = histc(val,spec_cdf);

if (bin ~= numel(spec_cdf))
    
  % linear interpolation
  slope = (spec_cdf(bin+1)-spec_cdf(bin))./(s(bin+1)-s(bin));
  intercept = spec_cdf(bin) - slope.*s(bin);
  invspec = (val - intercept)./slope;
  
else
    
  invspec = s(end);
  
end

