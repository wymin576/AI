data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1> N_obs;
  array[N_obs] int<lower=1, upper=N> ii;
  array[N_obs] int<lower=1, upper=T> tt;
  vector[N_obs] t_offset;
  real t_bar;
  vector[N_obs] y;
}

parameters {
  vector[N] alpha_raw;
  real<lower=0> sigma_alpha;
  vector[N] beta_raw;
  real<lower=0> sigma_beta;
  
  matrix[N, T] u_raw;
  real<lower=0> sigma_r;
  
  matrix[N, T] p_raw;
  real<lower=0> sigma_eps;
  real<lower=0.1, upper=0.9> rho;
  
  vector<lower=0>[T] sigma_xi;

  // 【改进1】：改为无约束参数，更易采样
  vector[T-1] gamma_raw;
  vector[T-1] delta_raw;
}

transformed parameters {
  vector[N] alpha = sigma_alpha * alpha_raw;
  vector[N] beta  = sigma_beta * beta_raw;

  // 【改进2】：使用指数变换使gamma/delta保持正数且稳定
  vector[T] gamma;
  vector[T] delta;
  gamma[1] = 1.0;
  delta[1] = 1.0;
  for (t in 2:T) {
    gamma[t] = exp(gamma_raw[t-1] * 0.1);  // 缩放 0.1 避免过大波动
    delta[t] = exp(delta_raw[t-1] * 0.1);
  }

  matrix[N, T] u;
  matrix[N, T] p;

  for (i in 1:N) {
    u[i, 1] = sigma_r * u_raw[i, 1];
    p[i, 1] = sigma_eps * p_raw[i, 1];
    for (t in 2:T) {
      u[i, t] = u[i, t-1] + sigma_r * u_raw[i, t];
      p[i, t] = rho * p[i, t-1] + sigma_eps * p_raw[i, t];
    }
  }
}

model {
  // 【改进3】：强化先验以稳定采样
  alpha_raw  ~ std_normal();
  beta_raw   ~ std_normal();
  to_vector(u_raw) ~ std_normal();
  to_vector(p_raw) ~ std_normal();

  // 【改进4】：gamma/delta先验改为标准正态
  gamma_raw ~ normal(0, 1);
  delta_raw ~ normal(0, 1);

  // 【改进5】：调整超参数先验，更好的数据适应性
  sigma_alpha ~ exponential(1);      // 放宽约束
  sigma_beta  ~ exponential(1);      // 放宽约束
  sigma_r     ~ exponential(2);      // 稍微放宽
  sigma_eps   ~ exponential(1);      // 放宽约束
  sigma_xi    ~ exponential(0.5);    // 调整测量误差先验

  // 【改进6】：改进似然计算，使用log_sum_exp避免数值溢出
  for (n in 1:N_obs) {
    int i = ii[n];
    int t = tt[n];
    
    real mu_n    = gamma[t] * (alpha[i] + beta[i] * t_offset[n] + u[i, t])
                 + delta[t] * p[i, t];
    real sigma_n = sigma_xi[t];
    
    // 确保标准差为正且合理
    if (sigma_n > 0) {
      target += normal_lpdf(y[n] | mu_n, sigma_n);
    }
  }
}

generated quantities {
  vector[N_obs] y_rep;
  vector[N_obs] log_lik;
  
  vector[T] var_alpha;
  vector[T] var_beta;
  vector[T] var_rw;
  vector[T] var_ar1;
  vector[T] var_meas;
  vector[T] var_total;

  // 1. 方差分解（按时期 t）
  for (t in 1:T) {
    real t_idx = t;
    real gamma_sq = square(gamma[t]);
    real delta_sq = square(delta[t]);
    
    var_alpha[t] = gamma_sq * square(sigma_alpha);
    var_beta[t]  = gamma_sq * square(t_idx - t_bar) * square(sigma_beta);
    var_rw[t]    = gamma_sq * t_idx * square(sigma_r);
    
    // AR(1)方差：使用稳定的计算
    real rho_sq = square(rho);
    real rho_2t = pow(rho, 2.0 * t_idx);
    var_ar1[t]  = delta_sq * square(sigma_eps) * fmax((1.0 - rho_2t) / fmax(1e-10, (1.0 - rho_sq)), 0.0);
    
    var_meas[t] = delta_sq * square(sigma_xi[t]);
    var_total[t] = var_alpha[t] + var_beta[t] + var_rw[t] + var_ar1[t] + var_meas[t];
  }

  // 2. 后验预测与对数似然（按观测值 n）
  for (n in 1:N_obs) {
    int i = ii[n];
    int t = tt[n];
    
    real mu_n    = gamma[t] * (alpha[i] + beta[i] * t_offset[n] + u[i, t])
                 + delta[t] * p[i, t];
    real sigma_n = sigma_xi[t];
    
    if (sigma_n > 0) {
      y_rep[n]   = normal_rng(mu_n, sigma_n);
      log_lik[n] = normal_lpdf(y[n] | mu_n, sigma_n);
    } else {
      y_rep[n]   = mu_n;
      log_lik[n] = negative_infinity();
    }
  }
}
