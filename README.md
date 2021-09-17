# Spiking NN


## How to Try
```
python src/snn.py
```

- 0-1 sec: Before learning
- 1-10 sec: During learning
- 10- sec: After learning


## FYI
### Izhikevich neuron model
Dynamics of the $i$-th neuron is as follows:
\[
    \begin{aligned}
    C \dot{v}_i &= k (v_i - v_\mathrm{r}) (v_i - v_\mathrm{t}) - u_i + I,\\
    \dot{u}_i &= a [b (v_i - v_\mathrm{r}) - u_i],
    \end{aligned}
\]
where $I$ is an input signal.
A spike is generated if $v_i \geq v_\mathrm{peak}$, and then the states are reset as follows:
\[
    \begin{aligned}
    v_i &\leftarrow v_\mathrm{reset}\\
    u_i &\leftarrow u_i + d.
    \end{aligned}
\]

Using
\[
    \dot{x}(t) \approx \frac{x(t + \Delta t) - x(t)}{\Delta t},
\]
the dynamics can be updated as follows:
\[
    \begin{aligned}
    v_i(t + \Delta t) &\approx v_i(t) + \frac{\Delta t}{C} [k (v_i(t) - v_\mathrm{r}) (v_i(t) - v_\mathrm{t}) - u_i(t) + I(t)],\\
    u_i(t + \Delta t) &\approx u_i(t) + a [b (v_i(t) - v_\mathrm{r}) - u_i(t)]\Delta t,
    \end{aligned}
\]


### Double exponential synaptic filter
Dynamics of the $j$-th neuron is as follows:
\[
    \begin{aligned}
    \dot{r}_j &= -\frac{r_j}{\tau_\mathrm{d}} + h_j,\\
    \dot{h}_j &= -\frac{h_j}{\tau_\mathrm{r}} + \frac{1}{\tau_\mathrm{r} \tau_\mathrm{d}} \sum_{k} \delta(t - t_{jk}),
    \end{aligned}
\]
where $t_{jk}$ indicates the time when the $k$-th spike occurred at the $j$-th neuron;
note that $t_{jk} < t$ is hold.

Using
\[
    \dot{x}(t) \approx \frac{x(t + \Delta t) - x(t)}{\Delta t},
\]
the dynamics can be updated as follows:
\[
    \begin{aligned}
    r_j(t + \Delta t) &\approx \left(1 - \frac{\Delta t}{\tau_\mathrm{d}}\right) r_j(t) + h_j(t) \Delta t,\\
    h_j(t + \Delta t) &\approx \left(1 - \frac{\Delta t}{\tau_\mathrm{r}}\right) h_j(t) + \frac{1}{\tau_\mathrm{r} \tau_\mathrm{d}} \sum_{k} \int_t^{t + \Delta t} \delta(t' - t_{jk}) \mathrm{d}t'.
    \end{aligned}
\]
By assuming that only at most one spike can occur during $[t, t + \Delta t)$, the latter equation can be modified as follows:
\[
    h_j(t + \Delta t) \approx
    \begin{cases}
    \left(1 - \frac{\Delta t}{\tau_\mathrm{r}}\right) h_j(t) + \frac{1}{\tau_\mathrm{r} \tau_\mathrm{d}}, & \text{if spike occurred},\\
    \left(1 - \frac{\Delta t}{\tau_\mathrm{r}}\right) h_j(t), & \text{if no spike occurred}.
    \end{cases}
\]


## References
- [[Nicola&Clopath, 2017](https://www.nature.com/articles/s41467-017-01827-3)] W. Nicola and C. Clopath, "Supervised learning in spiking neural networks with FORCE training," _Nature Communications_, vol. 8, no. 1, pp. 1-15, 2017.
- [FORCE法によるRecurrent Spiking Neural Networksの教師あり学習 - 知識のサラダボウル](https://omedstu.jimdo.com/2019/07/05/force%E6%B3%95%E3%81%AB%E3%82%88%E3%82%8Brecurrent-spiking-neural-networks%E3%81%AE%E6%95%99%E5%B8%AB%E3%81%82%E3%82%8A%E5%AD%A6%E7%BF%92/)
- [takyamamoto/SNN_FORCE-training: Spiking neural network with FORCE training](https://github.com/takyamamoto/SNN_FORCE-training)
