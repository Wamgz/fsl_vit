

'''

(batch, channels, height, width)
patch embedding -> (batch, num_patch * num_patch, embed_dim)
attention -> q: (batch, num_patch * num_patch, inner_dim), k: (batch, num_patch * num_patch, inner_dim) , v: (batch, num_patch * num_patch, inner_dim)
          -> q * k^T (batch, num_patch * num_patch, num_patch * num_patch)


'''