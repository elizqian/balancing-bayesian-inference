# Balanced truncation for Bayesian inference

This repository contains code for the numerical results of the following preprint:

1. Qian, E., Tabeart, J. M., Beattie, C., Gugercin, S., Jiang, J., Kramer, P. R., and Narayan, A.
[Model reduction for linear dynamical systems via balancing for Bayesian inference.](https://arxiv.org/abs/2111.13246)
To appear, Journal of Scientific Computing, 2022.<details><summary>BibTeX</summary><pre>
@article{Qian2021Balancing,
    title   = {Model reduction for linear dynamical systems via balancing for Bayesian inference},
    author  = {Qian, E. and Tabeart, J. M. and Beattie, C. and Gugercin, S. and Jiang, J. and Kramer, P. R. and Narayan, A.},
    journal = {arXiv:2111.13246},
    url     = {https://arxiv.org/abs/2111.13246},
    year    = {2022},
}</pre></details>

## Summary
The work in [1] considers a Bayesian approach to the task of inferring the unknown initial state of a linear dynamical system based on noisy linear measurements taken after the initial time. The initial state is endowed with a Gaussian prior and measurement noise is also assumed to be Gaussian, so that the Bayesian posterior is also Gaussian. We define a <b>balanced truncation for Bayesian inference</b> model reduction approach and show that the resulting reduced models inherit stability and error guarantees from the system-theoretic setting, and in some settings yield an optimal posterior covariance approximation as defined in [2].

## Examples
To generate the plots from the paper, run the *_plot{1,2}.m scripts, corresponding to:
* heat_plot1.m: The heat equation example with 500,000 measurements spaced 10^-4 seconds apart (Figure 5.1) -- this example may take a minute or two to run.
* heat_plot2.m: The heat equation example with 100 measurements spaced 0.1 seconds apart (Figure 5.2)
* iss_plot1.m: The ISS example with 3000 measurements spaced 0.1 seconds apart (Figure 5.3)
* iss_plot2.m: The ISS example with 10 measurements spaced 1 second apart (Figure 5.4)

## References
2. Spantini, A., Solonen, A., Cui, T., Martin, J., Tenorio, L., and Marzouk, Y. "[Optimal low-rank approximations of Bayesian linear inverse problems](https://epubs.siam.org/doi/pdf/10.1137/140977308?casa_token=CaYk5XimLkoAAAAA:-WjPu7U7kT8q3WZU66efl5X6GPylJOcnJM7XuOyy-I00LLa0vo9478Tv4BeNFoO67EwOsvl78Q)." SIAM Journal on Scientific Computing 37, no. 6 (2015): A2451-A2487.

### Contact
Please feel free to contact [Elizabeth Qian](http://www.elizabethqian.com/) with any questions about this repository or the associated paper.

### Acknowledgments
Thanks to Josie König for pointing out a bug.
