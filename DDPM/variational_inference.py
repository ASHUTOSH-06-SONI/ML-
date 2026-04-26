"""
So Variational infernece is what we hybridize our markov chain with 
how it works?
variational inference is basically a technique to approximate the integrals 
why ? coz its too many combinations to brute force on a computer, those gpu's ain't cheap son.
> we basically need to find p(z/x) but z is hidden and p(x) is found via these integrations (over all possible z)
so instead, q(z) ~= p(z/x)
> q is selected from any one of those sample distributions like normal, exponential and so on 
> this distribution that we select, it isn't exactly as out true distribution, so we just try to minimize the gap
> this minimized gap is called the KL Divergence
Dkl = Expectation<under q>(log(q(z)/p(z/x)))
lol we still need p(z/x)
so now we come up with something called ELBO, Evidence Lower Bound
won't derive it but the experession boils down to something like this
log(p(x)) = Dkl + Expectation<under q>(p(x and z)) - Expectation<under q>(log(q(z))
"""
import torch
import torch.optim.adam
mu = torch.nn.Parameter(torch.tensor(0.0))
log_sigma = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam([mu,log_sigma],lr=0.01)
for step in range(1000):
    sigma = torch.exp(log_sigma)
    elbo = 0
    k = 20  
    for _ in range(k):
        eps = torch.randn(())
        z = mu + sigma * eps
        log_pz = -0.5 * z**2
        log_px_given_z = -0.5 * (2.0 - z)**2
        log_q = -0.5 * ((z - mu)/sigma)**2 - log_sigma
        elbo += (log_pz + log_px_given_z - log_q)
    elbo = elbo / k
    loss = -elbo
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step}, ELBO: {elbo.item()}, mu: {mu.item()}, sigma: {sigma.item()}")
print(elbo)
print(torch.exp(log_sigma).item())
print(mu.item())

"""
what we did just now was VI for a single latent variable
now if we do this for multiple hidden variables which are rather chains
so 
"""