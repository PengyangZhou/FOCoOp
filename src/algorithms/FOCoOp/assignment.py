import matplotlib.pyplot as plt
import ot
import torch


def kmeans_pytorch(data, num_clusters=2, num_iterations=100):
    centers = data[torch.randperm(data.size(0))[:num_clusters]]

    for _ in range(num_iterations):

        distances = torch.cdist(data, centers, p=2)

        cluster_assignments = distances.argmin(dim=1)

        new_centers = torch.stack(
            [
                data[cluster_assignments == i].mean(dim=0) if (cluster_assignments == i).sum() > 0 else centers[i]
                for i in range(num_clusters)
            ]
        )

        if torch.allclose(centers, new_centers):
            break
        centers = new_centers

    return cluster_assignments, centers


if __name__ == "__main__":
    BN = 80
    D = 512
    NC = 6
    BN_o = 20
    tau = 0.0002
    mu = 0.5
    #  note step1 generate samples and ot mapping
    mean1 = torch.randn(D)
    cov1 = torch.eye(D) + 0.1 * torch.rand(D, D)
    cov1 = (cov1 + cov1.T) / 2
    features1 = torch.distributions.MultivariateNormal(mean1, cov1).sample((BN + NC,))

    low2 = 0.0
    high2 = 10.0
    features2 = torch.empty(BN_o, D).uniform_(low2, high2)

    C = features1[-NC:, :]
    Z = torch.cat((features1[:BN], features2), dim=0)

    cost = ot.dist(Z, C)
    cost /= cost.max()
    m = cost.shape[0]
    n = cost.shape[1]
    p_x = torch.ones(m) / m
    p_y = torch.ones(n) / n
    UOT_Pi = ot.unbalanced.mm_unbalanced(p_x, p_y, cost, reg_m=0.01, div="kl")
    print(UOT_Pi)

    prob = UOT_Pi.sum(dim=1, keepdim=True)

    cluster_assignments, centers = kmeans_pytorch(prob)

    sorted_indices = torch.argsort(centers.squeeze())
    new_labels_map = {old: new for new, old in enumerate(sorted_indices.tolist())}
    adjusted_labels = torch.tensor([new_labels_map[label.item()] for label in cluster_assignments])

    id_prob_min, in_prob_indixes = torch.topk(prob.squeeze(), 10, largest=False)
    n_remaining_ood = (adjusted_labels == 0).sum()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        prob.flatten(), torch.zeros_like(prob).flatten(), c=cluster_assignments, cmap="viridis", label="Clusters"
    )
    plt.scatter(centers.flatten(), torch.zeros_like(centers).flatten(), color="red", marker="x", label="Centers")
    plt.title("1D Data Clustering with K-means (PyTorch)")
    plt.xlabel("Data Points")
    plt.ylabel("Cluster Assignment")
    plt.legend()
    plt.show()

    filter_PI = UOT_Pi[adjusted_labels == 1, :]
    filter_Z = Z[adjusted_labels == 1, :]

    C_new = mu * C + (1 - mu) * (filter_PI.T @ filter_Z)

    cost = ot.dist(C, C_new)
    cost /= cost.max()
    m = cost.shape[0]
    n = cost.shape[1]
    p_x = torch.ones(m) / m
    p_y = torch.ones(n) / n
    after_Pi = ot.unbalanced.mm_unbalanced(p_x, p_y, cost, reg_m=0.1, div="kl")
    print(after_Pi)
    print("done")
