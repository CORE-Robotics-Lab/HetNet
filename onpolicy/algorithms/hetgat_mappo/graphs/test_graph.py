import dgl
import torch

def test_build_heterograph():
    data_dict = {
    ('user', 'follows', 'user'): (torch.tensor([0, 6, 2]), torch.tensor([1, 2, 1])),
    ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
    ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
    }
    num_nodes_dict = {'user': 6, 'topic': 4, 'game': 6}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    # g = dgl.heterograph(data_dict)
    print(g)
    pass

if __name__ == '__main__':
    test_build_heterograph()