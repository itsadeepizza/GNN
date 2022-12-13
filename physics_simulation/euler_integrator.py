import torch

def integrator(position, acc):
    """
    Calculate new position and speed using acceleration (useful for testing multiple steps)
    position_i = (p_i ^ t(k - C), , ..., p_i ^ t(k-1), p_i ^ tk)
    acc = (p_i ^ tk'')

    last_position_i =  p_i ^ t(k + 1)

    """
    second_last_position = position[:, -1, :]
    second_last_speed = second_last_position - position[:, -2, :]
    last_speed = second_last_speed + acc
    last_position = second_last_position + last_speed
    return last_position

def get_acc(position, last_position, normalization_stats=None):
    """
    Calculate ground truth acceleration using current  and next position (useful for calculating
    loss in train)
    position_i = (p_i ^ t_{k - C}, , ..., p_i ^ t_{k-1}, p_i ^ t_k)
    labels = p_i ^ t_{k+1}

    return acc =  p_i ^ tk''

    """

    def normalize(acceleration, normalization_stats):
        if normalization_stats is None:
            return acceleration

        return (acceleration) / \
               normalization_stats['acceleration']['std']

    second_last_position = position[:, -1, :]
    last_speed = last_position - second_last_position
    second_last_speed = second_last_position - position[:, -2, :]
    acc = last_speed - second_last_speed
    if normalization_stats==None:
        return acc
    return normalize(acc, normalization_stats)


if __name__ == "__main__":
    N = 2
    position = torch.randn(N, 6, 2)
    acc = torch.randn(N, 2)
    print("Pos:", position)
    print("Acc:", acc)
    last_position = integrator(position, acc)
    print(last_position)