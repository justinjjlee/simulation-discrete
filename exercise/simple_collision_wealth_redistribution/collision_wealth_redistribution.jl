# Load package
using Pkg;
using Distributions, Statistics, LinearAlgebra
using Plots

# Define population, N()
n_pop = 2_500; # Population size
n_iter = 5000; # Iteration of collision

# Distribution of income
μ_income = 50_000; # Mean income
σ_income = 12_000; # Standard deviation of income
𝐃 = Normal(μ_income, σ_income)

# Decider for selecting two agents to collide
𝐔 = Uniform(1, n_pop)

# vector of initial income of individual
val_inc_ini = deepcopy(rand(𝐃, n_pop));
# Equal splits
val_inc_trc = zeros(n_pop, n_iter); # Trace the changes in distribuiton
val_inc_trc[:,1] = val_inc_ini; # the very first distributional look
# Uniform random split
val_inc_trc_rand = zeros(n_pop, n_iter); # Trace the changes in distribuiton
val_inc_trc_rand[:,1] = val_inc_ini; # the very first distributional look


# Agent collide-split economy: two agents meet at a time, their wealth is equally distributed

for iter ∈ 2:n_iter
    # initialize the starting wealth distributio of the period
    val_inc_trc[:, iter] = val_inc_trc[:, (iter - 1)];
    # For each iteration, we select two agents who will collide
    agt_1, agt_2 = sample(1:n_pop, 2; replace = false)
    # collect their financial values
    val_inc_agt_1, val_inc_agt_2 = val_inc_trc[agt_1, iter], val_inc_trc[agt_2, iter];
    # divide up their wealth equally, and save
    val_inc_distributed = mean([val_inc_agt_1, val_inc_agt_2]);
    val_inc_trc[agt_1, iter] = val_inc_distributed;
    val_inc_trc[agt_2, iter] = val_inc_distributed;

    # Iteration 2 - uniform wealth redistribution, ranom split =========================
    # initialize the starting wealth distributio of the period
    val_inc_trc_rand[:, iter] = val_inc_trc_rand[:, (iter - 1)];
    # For each iteration, we select two agents who will collide
    agt_1, agt_2 = sample(1:n_pop, 2; replace = false)
    # collect their financial values
    total_wealth = val_inc_trc_rand[agt_1, iter] + val_inc_trc_rand[agt_2, iter];
    # How to split the wealth ..........................................................
    ρ = rand(1)[1]; #proportion to be split`
    # Redistribute wealth
    #   because of randomization, no need for random in order,
    val_inc_agt_1, val_inc_agt_2 = total_wealth * ρ, total_wealth * (1 - ρ)
    val_inc_trc_rand[agt_1, iter] = val_inc_agt_1;
    val_inc_trc_rand[agt_2, iter] = val_inc_agt_2;
end

println("Mean - true: $(μ_income); std - true: $(σ_income)")
println("Mean: $(mean(val_inc_trc[:,end])); std: $(std(val_inc_trc[:,end]))")
@gif for i ∈ 1:10:n_iter
    histogram(val_inc_trc[:,i], title = "Iteration: $(i)", xlims = (0,1e5))
end every 5

println("Mean - true: $(μ_income); std - true: $(σ_income)")
println("Mean: $(mean(val_inc_trc_rand[:,end])); std: $(std(val_inc_trc_rand[:,end]))")
@gif for i ∈ 1:10:n_iter
    histogram(val_inc_trc_rand[:,i], title = "Iteration: $(i)", xlims = (0,2e5))
end every 5
