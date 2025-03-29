import torch
import torch.distributions as dist


class TensorFIFOQueue:
    def __init__(self, capacity, feature_dim):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.queue = torch.empty((0, feature_dim)).cuda()

    def push(self, tensor):

        if tensor.ndim != 2 or tensor.shape[1] != self.feature_dim:
            raise ValueError(f"Tensor must have a shape of [N, {self.feature_dim}]")

        num_new_elements = tensor.size(0)
        num_available_spots = self.capacity - self.queue.size(0)

        if num_new_elements <= num_available_spots:
            self.queue = torch.cat((self.queue, tensor), dim=0)
        else:

            num_elements_to_remove = num_new_elements - num_available_spots
            self.queue = torch.cat((self.queue[num_elements_to_remove:], tensor), dim=0)

    def pop(self, num_elements):

        if self.queue.shape[0] < num_elements:
            raise ValueError(f"Not enough elements in the queue to pop {num_elements}")
        popped_elements = self.queue[:num_elements]
        self.queue = self.queue[num_elements:]
        return popped_elements

    def __len__(self):
        return self.queue.shape[0]

    def is_full(self):
        return self.queue.shape[0] == self.capacity


class ClassTensorQueues:
    def __init__(self, class_num, capacity, feature_dim):
        self.class_num = class_num
        self.queues = {class_id: TensorFIFOQueue(capacity, feature_dim) for class_id in range(class_num)}
        self.mean = {class_id: torch.empty((0, feature_dim)).cuda() for class_id in range(class_num)}
        self.covariance_matrix = {class_id: torch.empty((0, feature_dim)).cuda() for class_id in range(class_num)}

    def push(self, tensors, labels):

        if tensors.ndim != 2 or labels.ndim != 1 or tensors.shape[0] != labels.shape[0]:
            raise ValueError("The dimensions of tensors and labels do not match.")

        for tensor, label in zip(tensors, labels):
            if label < 0 or label >= self.class_num:
                raise ValueError(f"Label must be between 0 and {self.class_num - 1}")
            self.queues[label.item()].push(tensor.unsqueeze(0))

    def is_full(self):

        return all(queue.is_full() for queue in self.queues.values())

    def update_guassian(self):
        for class_id in range(self.class_num):
            self.mean[class_id] = torch.mean(self.queues[class_id].queue, 0)
            vectors_centered = self.queues[class_id].queue - self.mean[class_id]
            self.covariance_matrix[class_id] = vectors_centered.t().matmul(vectors_centered) / (
                vectors_centered.size(0) - 1
            )
            epsilon = 1e-8
            self.covariance_matrix[class_id] += epsilon * torch.eye(self.covariance_matrix[class_id].size(0)).cuda()
        self.multivariate_normal_dist = {
            class_id: dist.MultivariateNormal(self.mean[class_id], self.covariance_matrix[class_id])
            for class_id in range(self.class_num)
        }

    def sampling_guassian(self, num_samples=1000, selected_number=10, soft_y=0, soft_split=False):
        if soft_split:

            syn_x_list = []
            syn_y_list = []
            id_label = []
            for class_id in range(self.class_num):
                multivariate_normal_dist = self.multivariate_normal_dist[class_id]
                new_samples = multivariate_normal_dist.sample((num_samples,))

                log_probabilities = multivariate_normal_dist.log_prob(new_samples)
                value, indice = torch.sort(log_probabilities)
                syn_x_list.append(new_samples[indice])

                gradual_weights = torch.arange(num_samples).view(-1, 1).cuda() / num_samples
                soft_ood_y = soft_y.new_zeros((num_samples, soft_y.size(1)))
                soft_ood_y[:, -1] = 1
                soft_id_y = soft_y.new_zeros((num_samples, soft_y.size(1)))
                soft_id_y[:, class_id] = 1
                syn_y_cate = gradual_weights * soft_ood_y + (1 - gradual_weights) * soft_id_y
                syn_y_list.append(syn_y_cate)
            syn_x = torch.cat(syn_x_list, dim=0)
            syn_y = torch.cat(syn_y_list, dim=0)
        else:

            id_feat = []
            id_label = []
            ood_feat = []
            for class_id in range(self.class_num):
                multivariate_normal_dist = self.multivariate_normal_dist[class_id]
                new_samples = multivariate_normal_dist.sample((num_samples,))
                log_probabilities = multivariate_normal_dist.log_prob(new_samples)

                value, indice = torch.sort(log_probabilities)
                id_syn = new_samples[indice[-selected_number:]]
                ood_syn = new_samples[indice[:selected_number]]
                id_feat.append(id_syn)
                id_label.append(torch.ones(selected_number).cuda() * class_id)
                ood_feat.append(ood_syn)

            id_feat = torch.cat(id_feat, dim=0)
            ood_feat = torch.cat(ood_feat, dim=0)
            id_label = torch.cat(id_label, dim=0)

            syn_x = torch.cat([id_feat, ood_feat], dim=0)
            soft_id_y = torch.zeros((id_feat.size(0), soft_y.size(1)), dtype=soft_y.dtype, device=soft_y.device)
            soft_id_y[torch.arange(id_feat.size(0)).cuda(), id_label.long().cuda()] = 1
            soft_ood_y = soft_y.new_zeros((ood_feat.size(0), soft_y.size(1)))
            soft_ood_y[:, -1] = 1
            syn_y = torch.cat([soft_id_y, soft_ood_y], dim=0)
        return syn_x, syn_y
