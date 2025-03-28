def denormalize(x, mean, std):
    x = x * std
    x = x + mean
    return x