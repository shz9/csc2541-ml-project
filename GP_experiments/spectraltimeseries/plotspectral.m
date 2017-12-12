% compare empirical spectral density with learned SM spectral density

[emp lemp s] = empspect(ytrain);
sfine = [s(1):1e-4:s(end)]';
[sm_spect] = smspect(sfine,smhyp_train.cov,Q);

% look at the log spectral densities
figure(2); clf; hold on;
plot(s,lemp,'LineWidth', 2, 'Color',[0.494117647058824 0.184313725490196 0.556862745098039]); hold on;
plot(sfine, log(sm_spect), 'LineWidth', 2, 'Color', 'k');
xlabel('frequency');
ylabel('log spectral density');
