Modeling Uncertainty in RNNs for Time Series Forecasting
====
CSC2541 | University of Toronto | Fall 2017

### Group Members:
- George-Alexandru Adam
- Shadi Zabad
- Tahmid Mehdi
- Abhishek Tiwari

### Abstract:

Modeling uncertainty in neural networks has recently become a central theme of
much fruitful research in the machine learning literature. This surge of interest
is partially motivated by the crucial role that uncertainties play in many practical
applications, especially in areas such as Active and Reinforcement Learning. Despite
this flurry of research, there has been relatively little work done on modeling
uncertainty in Recurrent Neural Networks (RNNs) in the context of time series
forecasting. RNNs present a unique challenge due to their optimization paradigm
known as Backprop through time. A couple of approaches to representing uncertainty
in RNNs have been proposed, but the analysis done in the corresponding
publications is limited to epistemic uncertainty. We instead focus on three main
types of uncertainty: optimization uncertainty, architecture uncertainty, and data
uncertainty. The approach we advocate is quite general and can be used for domains
outside of time series analysis. It may also be easily extended to other neural
network architectures.


### Experimental Design:

![Alt text](/md_figures/flowchart.png?raw=true "Flow chart")


Figure 2 shows the general approach we employ to model various types of uncertainty in RNNs.
To model the 3 dimensions of uncertainty discussed below, we create an ensemble of RNNs that
are trained with different weight initializations, different hyperparameter settings, or by using
bootstrapped samples of the data. This allows us to model uncertainty in the optimization procedure,
architecture, and data respectively.

![Alt text](/md_figures/algorithm.png?raw=true "Algorithm")

Algorithm 1 describes the general procedure for how we use ensembles of networks to obtain
uncertainty estimates. Depending on the ensemble type, either the data, initial weights, or network
hyperparameters become a list of different settings, while the other arguments stay the same. For
example, if the ensemble type is "bootstrap", then the data becomes a list of B different bootstrap samples, while the initial weights are randomly chosen and the hyperparameters are those determined
to be the best via cross-validation. This results in B different models being trained on various
bootstrap samples of the data. Alternatively, if the ensemble type is "initial weights", B different
models are trained using various random initializations of the weights one the entire training dataset
and with a fixed set of optimal hyperparameters. Lastly, if the ensemble type is "hyperparameters", B
different models are trained using a list of various hyperparameter settings, each of which provides
reasonable validation error. In all three cases, we take the resulting B models, make predictions on
the test set, and use those predictions to obtain a mean and standard deviation.

### Results:

![Alt text](/md_figures/gp.png?raw=true "GP Results")

![Alt text](/md_figures/weight_init.png?raw=true "Weight Initializations Results")

![Alt text](/md_figures/bootstrap_samples.png?raw=true "Bootstrap Samples Results")

![Alt text](/md_figures/architectures.png?raw=true "Architectures Results")

### Conclusion:

Time series data is a great fit for RNNs due to their ability to model long-term dependencies. The
flexibility offered by RNNs spares data scientists from having to tinker with kernels for Gaussian
processes. Although GPs outperformed RNN models on the Lake Erie dataset, this is likely due to
information leak from the test set when designing the kernel. This shows the need for a data-driven
time series modeling approach that is not plagued by human subjectivity, and is thus more consistent.
We have added further benefits to using RNNs for time series forecasting by showing how to obtain
reasonable confidence intervals for their predictions through straightforward training of multiple
models. We explored three different sources of uncertainty, each of which provides insight into
diverse aspects of RNN training and model selection. Such techniques can be transferred to other
neural network architectures, and can help analyze the stability of regression models without using
Bayesian inference.
