# Deriving the local optimality condition for $S_2$

This note fills in the derivation behind the one-line condition in
[`findings.md` §8](findings.md#8-answers):

$$
\frac{\left|\partial T/\partial m_1\right|}
     {\left|\partial T/\partial m_2\right|}
= \frac{\beta_1}{\beta_2}
= \frac{c_1}{c_2}r.
$$

The condition is the usual equal-marginal-value rule for splitting a fixed amount of
money between two servers. It does not depend on the detailed fork-join approximation;
that approximation supplies the two partial derivatives that appear in the rule.

## 1. From capacities to effective service rates

Let the two servers' unscaled service rates be

$$
\widehat\mu_1,
\qquad
\widehat\mu_2=r\widehat\mu_1,
$$

and let their capacity multipliers be $S_1$ and $S_2$. Their effective service rates
are then

$$
m_1=\widehat\mu_1S_1,
\qquad
m_2=\widehat\mu_2S_2=r\widehat\mu_1S_2.
$$

If one unit of capacity costs $c_1$ or $c_2$, the station's total spend is

$$
\begin{aligned}
B
  &=c_1S_1+c_2S_2 \\
  &=\frac{c_1}{\widehat\mu_1}m_1
    +\frac{c_2}{r\widehat\mu_1}m_2 \\
  &=\beta_1m_1+\beta_2m_2,
\end{aligned}
$$

where

$$
\beta_1=\frac{c_1}{\widehat\mu_1},
\qquad
\beta_2=\frac{c_2}{r\widehat\mu_1}.
$$

Thus $\beta_k$ is the price of adding one unit to server $k$'s effective service
rate. At a fixed spend $B$, all affordable choices lie on the line

$$
\beta_1m_1+\beta_2m_2=B.
$$

## 2. The constrained optimization problem

For a station with arrival rate $\lambda$, let $T(m_1,m_2)$ denote its predicted mean
response time. The local allocation problem is

$$
\begin{aligned}
\text{minimize}\quad &T(m_1,m_2),\\
\text{subject to}\quad &\beta_1m_1+\beta_2m_2=B,\\
&m_1>\lambda,\quad m_2>\lambda.
\end{aligned}
$$

The last two inequalities are the stability requirements. A stable split exists only when

$$
B>\lambda(\beta_1+\beta_2).
$$

## 3. A spend-preserving exchange

Suppose we make a small change $dm_1$ while keeping the station spend fixed. Differentiating
the budget constraint gives

$$
\beta_1\,dm_1+\beta_2\,dm_2=0,
$$

and hence

$$
dm_2=-\frac{\beta_1}{\beta_2}dm_1.
$$

This describes a small budget transfer: increasing $m_1$ requires reducing $m_2$ by
exactly enough to pay for it. The resulting first-order change in response time is

$$
\begin{aligned}
dT
  &=\frac{\partial T}{\partial m_1}dm_1
    +\frac{\partial T}{\partial m_2}dm_2 \\
  &=\left(
      \frac{\partial T}{\partial m_1}
      -\frac{\beta_1}{\beta_2}
       \frac{\partial T}{\partial m_2}
    \right)dm_1.
\end{aligned}
$$

At a smooth interior optimum, neither direction of this budget transfer may improve the
response time. Therefore the coefficient of $dm_1$ must be zero:

$$
\frac{\partial T}{\partial m_1}
=\frac{\beta_1}{\beta_2}
 \frac{\partial T}{\partial m_2}.
$$

Increasing either effective service rate normally decreases response time, so both partial
derivatives are negative. Taking their magnitudes gives

$$
\boxed{
\frac{\left|\partial T/\partial m_1\right|}
     {\left|\partial T/\partial m_2\right|}
=\frac{\beta_1}{\beta_2}
}
$$

and substituting the definitions of the prices yields

$$
\boxed{
\frac{\left|\partial T/\partial m_1\right|}
     {\left|\partial T/\partial m_2\right|}
=\frac{c_1/\widehat\mu_1}{c_2/(r\widehat\mu_1)}
=\frac{c_1}{c_2}r.
}
$$

Equivalently,

$$
\frac{\left|\partial T/\partial m_1\right|}{\beta_1}
=
\frac{\left|\partial T/\partial m_2\right|}{\beta_2}.
$$

Each side is the marginal reduction in response time obtained from one additional dollar.
If the left side is larger, moving money from server 2 to server 1 improves the allocation;
if the right side is larger, the reverse transfer improves it. At the optimum the two
benefits per dollar agree.

The same result follows from a Lagrangian,

$$
\mathcal L
=T(m_1,m_2)
+\nu(\beta_1m_1+\beta_2m_2-B),
$$

because stationarity gives

$$
\frac{\partial T}{\partial m_1}=-\nu\beta_1,
\qquad
\frac{\partial T}{\partial m_2}=-\nu\beta_2.
$$

Dividing these equations produces the same condition.

## 4. Recovering $r^*$ and $S_2$

Define the selected effective-rate ratio by

$$
r^*=\frac{m_2^*}{m_1^*}.
$$

For a proposed $r^*$, its intersection with the spend line is

$$
m_1(r^*)=\frac{B}{\beta_1+\beta_2r^*},
\qquad
m_2(r^*)=\frac{Br^*}{\beta_1+\beta_2r^*}.
$$

Consequently, $r^*$ is determined implicitly by the scalar equation

$$
\frac{
 \left|T_1\bigl(m_1(r^*),m_2(r^*)\bigr)\right|
}{
 \left|T_2\bigl(m_1(r^*),m_2(r^*)\bigr)\right|
}
=\frac{\beta_1}{\beta_2},
$$

where $T_k=\partial T/\partial m_k$. This generally has to be solved numerically; the
one-line condition is not a closed-form expression for $r^*$.

Finally,

$$
\frac{m_2}{m_1}
=\frac{r\widehat\mu_1S_2}{\widehat\mu_1S_1}
=r\frac{S_2}{S_1},
$$

so the optimal capacity relationship is

$$
\boxed{S_2=\frac{r^*}{r}S_1.}
$$

This contains both pre-existing policies:

- $r^*=r$ gives $S_2=S_1$, the invariant-capacity policy.
- $r^*=1$ gives $S_2=S_1/r$, the equal-effective-rate policy.

## 5. The derivatives for `t_ul`

The preceding argument applies to any differentiable response-time model. In this project,
the model is

$$
T=(1-\alpha)T_{\mathrm{UB}}+\alpha T_{\mathrm{bot}},
$$

with

$$
\alpha=\frac{\lambda}{8}\left(\frac{1}{m_1}+\frac{1}{m_2}\right),
$$

$$
T_{\mathrm{UB}}
=\frac{1}{m_1-\lambda}
 +\frac{1}{m_2-\lambda}
 -\frac{1}{m_1+m_2-2\lambda},
$$

and

$$
T_{\mathrm{bot}}
=\frac{1}{\min(m_1,m_2)-\lambda}.
$$

Away from $m_1=m_2$, its partial derivative with respect to $m_k$ is

$$
\frac{\partial T}{\partial m_k}
=
\frac{\partial\alpha}{\partial m_k}
  (T_{\mathrm{bot}}-T_{\mathrm{UB}})
+(1-\alpha)\frac{\partial T_{\mathrm{UB}}}{\partial m_k}
+\alpha\frac{\partial T_{\mathrm{bot}}}{\partial m_k},
$$

where, for $j\ne k$,

$$
\frac{\partial\alpha}{\partial m_k}
=-\frac{\lambda}{8m_k^2},
$$

$$
\frac{\partial T_{\mathrm{UB}}}{\partial m_k}
=-\frac{1}{(m_k-\lambda)^2}
 +\frac{1}{(m_1+m_2-2\lambda)^2},
$$

and

$$
\frac{\partial T_{\mathrm{bot}}}{\partial m_k}
=
\begin{cases}
-\dfrac{1}{(m_k-\lambda)^2}, & m_k<m_j,\\[6pt]
0, & m_k>m_j.
\end{cases}
$$

Substituting these two partial derivatives into the marginal-value condition determines the
locally optimal split. The implementation evaluates this condition directly and solves it by
bisection in [`qopt/forkjoin_policy.py`](../../qopt/forkjoin_policy.py).

## 6. What “locally optimal” means here

There are three qualifications to the one-line statement.

First, the argument optimizes the split **at a fixed station spend**. If $S_1$ were held
literally fixed while cost-free increases in $S_2$ were allowed, the model would normally
keep improving as $S_2$ increased, and there would be no finite optimum. Here “selecting
$S_2$” means deciding how the fixed spend is divided between both capacities.

Second, the equality is a necessary first-order condition at a smooth interior optimum. The
restricted objective must additionally be unimodal or have the appropriate derivative sign
change for this point to be the unique/global minimum. The probes reported in `findings.md`
found that behavior for the studied configurations.

Third, $T_{\mathrm{bot}}$ has a kink at $m_1=m_2$. If the optimum occurs there, ordinary
two-sided partial derivatives do not exist. Parameterizing the spend line by $m_1$, define

$$
f'(m_1)
=T_1-\frac{\beta_1}{\beta_2}T_2.
$$

The correct condition at the kink is the one-sided condition

$$
f'_-(m_1^*)\le 0\le f'_+(m_1^*).
$$

The bisection implementation handles this by allowing the derivative sign change to bracket
the kink. This is how an exactly equal-rate answer, $r^*=1$, can be selected.

## 7. Relation to the network optimizer

The derivation solves only the station's inner problem for a given spend $B$. In the full
network, equation 21 determines that spend, while the selected $r^*$ changes the station's
linear allocation price

$$
c_1+c_2\frac{r^*}{r}.
$$

The implementation therefore forms a nested fixed point: at each outer iteration it solves
the local condition at the current station spend, reprices the ray, and lets the network
allocator update the spend. Agreement with the network-level sweep is empirical validation
of that procedure; it does not follow from the local first-order condition alone.

There are now two such validations, and neither is implied by the derivation above. The
tuned policy reaches or beats what a $0.02$-grid sweep of $r^*$ finds in all three QCSC
workloads ([`implementation.md`](implementation.md)), and the resulting allocations were then
measured against a discrete-event simulation of the whole network: the predicted gains over
the incumbent ray survive, with each analytic gain landing inside its own measured $95\%$
interval, and the bias of $T$ at the tuned ray turns out
indistinguishable from its bias at the incumbent ray
([`simcheck-output.txt`](simcheck-output.txt)). That last point is what licenses using this
condition at a ray the approximation was never validated at — and it is a measurement, not a
consequence of the algebra.
