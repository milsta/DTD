# DTD
Distributed Consensus-Based Multi-Agent Temporal-Difference Learning

This is the repository of the code implementing the technical algorithms proposed in paper "Stanković, M. S., Beko, M., & Stanković, S. S. (2023). Distributed consensus-based multi-agent temporal-difference learning. Automatica, 151, 110922.".

The paper proposes two new distributed consensus-based algorithms for temporal-difference learning in multi-agent
Markov decision processes. The algorithms are of off-policy type and are aimed at linear approximation of the value function.
Restricting agents’ observations to local data and communications to their small neighborhoods, the algorithms consist of:
a) local updates of the parameter estimates based on either the standard TD(λ) or the emphatic ETD(λ) algorithm, and b)
dynamic consensus scheme implemented over a time-varying lossy communication network. The algorithms are completely
decentralized, allowing efficient parallelization and applications where the agents may have different behavior policies and
different initial state distributions while evaluating a common target policy.
