from . import kernel

# Spot check the day-of-week matrix
tmp = same_dow(np.arange(20), np.arange(20)).astype('int')
assert tmp[0, 0] == 1
assert tmp[0, 1] == 0
assert tmp[1, 1] == 1
assert tmp[7, 0] == 1
assert tmp[7, 1] == 0

dat = combined_kernel(np.arange(600), np.arange(600))
plot_matrix(dat[:100, :100], noticks=True)
_ = np.linalg.cholesky(dat)
