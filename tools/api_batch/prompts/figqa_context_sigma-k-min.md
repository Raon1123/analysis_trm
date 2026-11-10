A small weight-tied recurrent network (TRM) is trained on σ^k permutation-composition tasks: a
fixed permutation σ on n=10 elements is composed k times, and the task family is parameterized by
the composition depth k. Model variants ("cohorts") differ on two axes: whether they carry a latent
state z across supervision steps ("z" vs "noz"), and whether they run extra recurrent iterations
within a step ("iter" vs "noiter"). Each experimental cell is one training run at depth k with a
random seed s. A run is said to "collapse" when its test-exact accuracy ends at ≤ 0.5. The network
maintains up to two latent streams: z_L (low-level) and z_H (high-level). PR denotes participation
ratio, an effective-rank measure of a representation's covariance spectrum (larger PR = variance
spread over more dimensions).
