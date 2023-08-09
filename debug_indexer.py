from models.indexer_model import IndexerModule
import torch

import lovely_tensors as lt
lt.monkey_patch()

im = IndexerModule(
    lattice_size=50000,
    num_iterations=5,
    error_precision=0.0012,
    filter_precision=0.00075,
    filter_min_num_spots=6
)

initial_cell = torch.tensor([
    [78.3, 0., 0.],
    [0., 78.3, 0.],
    [0., 0., 37.96]
])

reciprocal_spots = torch.tensor([
    [-0.2740, 0.0319, 0.0440],
    [0.0142, -0.1314, 0.0099],
    [-0.0742, -0.2719, 0.0460],
    [0.0839, -0.0099, 0.0040],
    [-0.1303, -0.2077, 0.0346],
    [0.1650, -0.0412, 0.0165],
    [-0.0526, -0.0877, 0.0059],
    [0.0984, 0.1081, 0.0121],
    [-0.2149, 0.0320, 0.0270],
    [0.0421, -0.0937, 0.0060],
    [-0.0844, 0.1088, 0.0108],
    [-0.1012, 0.0388, 0.0066],
    [0.2421, 0.1052, 0.0402],
    [0.0258, 0.0735, 0.0034],
    [0.0420, 0.2995, 0.0531],
    [-0.0561, -0.0402, 0.0027],
    [-0.2467, -0.1653, 0.0512],
    [-0.1013, 0.0284, 0.0063],
    [-0.1430, -0.0305, 0.0121],
    [0.0715, -0.0967, 0.0082],
    [0.2114, -0.1623, 0.0410],
    [-0.0933, 0.1838, 0.0243],
    [0.0372, 0.2275, 0.0305],
    [0.1390, -0.1969, 0.0334],
    [-0.1088, 0.2650, 0.0475],
    [-0.0223, 0.2774, 0.0448],
    [-0.0937, -0.2198, 0.0328],
    [0.1032, 0.2682, 0.0478],
    [0.0218, -0.1213, 0.0086],
    [-0.2295, -0.2075, 0.0557],
    [0.0903, -0.3135, 0.0622],
    [0.0341, 0.0573, 0.0025],
    [-0.1653, -0.1234, 0.0243],
    [-0.1992, 0.1412, 0.0343],
    [-0.1745, 0.0341, 0.0180],
    [-0.0701, 0.0340, 0.0034],
    [-0.0242, -0.0830, 0.0042]
])

solution_successes, solution_bases, solution_indices, solution_errors, _ = im(
    reciprocal_spots.repeat(1, 1, 1),
    initial_cell,
    min_num_spots=8,
    angle_resolution=250,
    num_top_solutions=250
)

for success, basis, solution_mask, claimed_error in zip(
        solution_successes[0], solution_bases[0], solution_indices[0], solution_errors[0]
):
    print("Solution basis:")
    print(basis)
    print("Magnitud of solution basis(a, b, c):")
    print(basis.norm(dim=-1))
    miller_indices = torch.round(reciprocal_spots @ basis.T).int()
    print("Miller indices:")
    print((miller_indices * solution_mask[:, None]).p)

    break

