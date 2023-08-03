import torch
import torch.nn as nn
import math
from numpy import arange, pi, sin, cos, arccos
from torch.nn.functional import normalize
import sys


def batched_invert_matrix(A):
    """
        Inverts a batch of 3x3 matrices and returns NaN if any matrix is not invertible.

        Args:
            matrix_batch (torch.Tensor): Input tensor of shape (n, 3, 3) representing n 3x3 matrices.

        Returns:
            torch.Tensor: Output tensor of shape (n, 3, 3) representing the inverted matrices with NaN for non-invertible ones.
        """
    M = A.reshape(-1, 9)
    a = M[:, 0]
    b = M[:, 1]
    c = M[:, 2]
    d = M[:, 3]
    e = M[:, 4]
    f = M[:, 5]
    g = M[:, 6]
    h = M[:, 7]
    i = M[:, 8]

    coefficent = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    entries = []
    entries.append(e * i - f * h)
    entries.append(c * h - b * i)
    entries.append(b * f - c * e)
    entries.append(f * g - d * i)
    entries.append(a * i - c * g)
    entries.append(c * d - a * f)
    entries.append(d * h - e * g)
    entries.append(b * g - a * h)
    entries.append(a * e - b * d)
    entries = torch.stack(entries, dim=1)
    for k in range(9):
        entries[:, k] /= coefficent
    entries = entries.reshape(-1, 3, 3)
    return entries


def rotations(rodrigues_vector, alpha, angle_resolution: int):
    """
    This function uses the Rodrigues forumla to create rotations around the given rodrigues_vector with the given angle alpha
    @param rodrigues_vector: The vector around which to create the rotation matrices
    @param alpha: The angle od the rotation
    @param angle_resolution: The number of samples to be generated.
    @return: A tensor of bs, angle_resolution, 3, 3 containing bs * angle resolution rotation matrices, each rotating the corresponding alpha degrees given
    """
    ip = normalize(rodrigues_vector)
    new_ip = ip.flip(1)
    new_ip[:, 1] *= -1
    A = to_skew_symmetric(new_ip).permute(0, 2, 1).to(ip.device)
    AA = torch.bmm(A, A)
    first = torch.diag(torch.ones(3)).to(ip.device)
    second = A.repeat(angle_resolution, 1, 1, 1) * torch.sin(alpha)[:, None, None, None]
    third = AA.repeat(angle_resolution, 1, 1, 1) * (1 - torch.cos(alpha))[:, None, None, None]
    R = first.expand(second.shape) + second + third
    return R.permute(1, 0, 2, 3)


def batched_compute_angles(bases):
    """
    Computes the angles between the vectors of 3x3 bases. this is use to compute the quality of solution bases.
    @param bases: A tensor of shape bs, 3, 3 with bs bases
    @return: The angles alpha, beta and gamma between the consecutive vectors.
    """
    bases = torch.nn.functional.normalize(bases, dim=-1)
    alpha = torch.acos(torch.sum(bases[:, 0] * bases[:, 1], dim=-1)) * 180 / math.pi
    beta = torch.acos(torch.sum(bases[:, 0] * bases[:, 2], dim=-1)) * 180 / math.pi
    gamma = torch.acos(torch.sum(bases[:, 1] * bases[:, 2], dim=-1)) * 180 / math.pi
    return torch.stack([alpha, beta, gamma], 1)


def compute_penalization(matrix, initial_cell):
    """
    Computes de penalization used to skew the solutions to be as close as possible to the given ideal basis.
    @param matrix: The matrix to be tested
    @param initial_cell: the ideal basis
    @return: A penalization score that depends on how much matrix deviates from the structure of the initial_cell
    """
    device = matrix.device
    diff_cell = torch.abs(
        matrix.norm(dim=-1, p=2) - initial_cell[:3].norm(dim=-1, p=2).to(device)
    )
    total_diff_cell = torch.max(diff_cell, dim=-1).values.to(device)
    penalization = torch.max(torch.tensor(0).to(device), total_diff_cell) ** 3

    # we also penalize according to angle discrepancies
    angles = batched_compute_angles(matrix)  # n x 3
    angles_diff = torch.abs(
        angles - batched_compute_angles(initial_cell.unsqueeze(0))[0].to(device))
    total_diff_angles = torch.max(angles_diff, dim=-1).values.to(device)
    penalization += torch.max(torch.tensor(0).to(device), total_diff_angles) ** 3
    return penalization


def batched_subset_from_indices(input, indices):
    """
    Takes a subset of the input indicated by the indices in each batch
    @param input: The batched input with structure (bs, -1, shape_data) , e.g. (bs, 10000, 3, 3)
    @param indices: The batched indices that you want to take from each batch (bs, size_subset), e.g, (bs, 200)
    @return: Returns the subset of the input corresponding to he indices in each batch (bs, size_subset, shape_data)
    """
    bs = input.shape[0]
    size_subset = indices.shape[1]
    shape_data = list(input.shape[2:])
    numel = input[0, 0].numel()
    shape_output = [bs, size_subset] + shape_data
    return torch.gather(input.reshape(bs, input.shape[1], -1), 1, indices.repeat(numel, 1, 1).permute(1, 2, 0)).reshape(
        shape_output)


def create_sphere_lattice(num_points: int = 1000000):
    """
    Samples num_points vectors from the unit sphere using the golden ratio spiral.
    @param num_points: The number of vectors to be sampled
    @return: A collection of sampled vectors.
    """
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, num_points)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / num_points)
    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);
    lattice = torch.randn(num_points, 3)
    lattice[:, 0] = torch.FloatTensor(x)
    lattice[:, 1] = torch.FloatTensor(y)
    lattice[:, 2] = torch.FloatTensor(z)
    return lattice[lattice[:, -1] >= 0]


def to_skew_symmetric(x):
    """
    Takes a tensor bs, 3 of vectors in R^3 and creates the skew symmetric matrix corresponding to them.
    @param x: The input vector
    @return: The tensor bs, 3, 3 with the skew symmetric matrices corresponding to x
    """
    out = torch.zeros(x.shape[0], 3, 3).to(x.device)
    out[:, 0, 1] = x[:, 0]
    out[:, 0, 2] = x[:, 1]
    out[:, 1, 0] = -x[:, 0]
    out[:, 1, 2] = x[:, 2]
    out[:, 2, 0] = -x[:, 1]
    out[:, 2, 1] = -x[:, 2]
    return out


def rotation_to_target(sources, targets):
    """
    Construct rotation matrices in SO3 that map the vectors of source to those of tarhets, i.e.
    It takes s and t as input and creates a rotaiton matrix R such that Rs = t
    @param sources: a tensor bs, 3 with sources
    @param targets: a tensor bs, 3 with targets
    @return: a tensor bs, 3, 3 with the corresponding rotation matrices
    """
    sources = normalize(sources, dim=-1)
    targets = normalize(targets, dim=-1)
    V = torch.cross(sources, targets)
    SV = torch.zeros_like(V).to(sources.device)
    SV[:, 0] = -V[:, 2]
    SV[:, 1] = V[:, 1]
    SV[:, 2] = -V[:, 0]
    S = V.norm(dim=-1, p=2)
    C = torch.diag(torch.mm(sources, targets.T))
    SS = to_skew_symmetric(SV)
    SS2 = torch.bmm(SS, SS)
    I = torch.stack(sources.shape[0] * [torch.diag(torch.ones(3))], 0).to(sources.device)
    R = I + SS + ((1 - C) / S ** 2)[:, None, None] * SS2
    return R


def bcompute_penalization(X, Y):
    """
    Batched version of compute_penalization
    @param X: batched inputs
    @param Y: batched inputs
    @return: batched output
    """
    first = int(X.shape[0])
    second = int(X.shape[1])
    result = compute_penalization(X.flatten(0, 1), Y.flatten(0, 1))
    return result.unflatten(0, [first, second])


def bb_inverse(X):
    """
    Batched version of batched_invert_matrix
    @param X: batched inputs
    @return: batched output
    """
    first = int(X.shape[0])
    second = int(X.shape[1])
    result = batched_invert_matrix(X.flatten(0, 1))
    return result.unflatten(0, [first, second])


class IndexerModule(nn.Module):
    def __init__(self, lattice_size, num_iterations: int = 5, error_precision: float = 0.0015,
                 filter_precision: float = 0.00075, filter_min_num_spots: int = 6):
        super(IndexerModule, self).__init__()

        self.unite_sphere_lattice = torch.nn.Parameter(create_sphere_lattice(num_points=lattice_size))
        self.filter_precision = float(filter_precision)
        self.filter_min_num_spots = int(filter_min_num_spots)
        self.error_precision = float(error_precision)
        self.num_iterations = int(num_iterations)
        self.solution_sources = torch.zeros(0)
        gettrace = getattr(sys, 'gettrace', None)
        self.debugging = gettrace is not None

    def sample_bases(
            self,
            source,
            unite_sphere_lattice,
            initial_cell: torch.Tensor,
            dist_to_integer: float = 0.2,
            num_top_solutions: int = 1000,
            angle_resolution: int = 360
    ):
        """
        Computes the bases that are going to be candidates for solutions.
        Takes the first vector from the unite_sphere_lattice scaled by the length of the first vefctor of the initial_cell,
        and filters possible solutions by looking at those that have more integer projections.
        Then it attaches to each of those a copy of the initial_cell, and rotataes them angle_resolution times by keeping the first vector fixed.
        In this way it obtains a set of candidate bases
        @param source: The spots as 3D points projected on the Ewald sphere
        @param unite_sphere_lattice: A sample of points on the unit sphere
        @param initial_cell: The given initial basis cell for this molecule
        @param dist_to_integer: The maximum allowed distance for a projection to its closest integer to be considered as a true spot
        @param num_top_solutions: the number of vectors from the scaled unite_sphere_lattice that will be considered for the second part of the algorithm
        @param angle_resolution: The number of initial_cell that will be attached to each of the candidate vectors in the first direction.
        @return: A tensor of the form kx3x3, containing the top k bases, where k = num_top_solutions
        """

        device = source.device
        bs = int(source.shape[0])
        scaling = initial_cell.norm(dim=-1, p=2).to(device)

        # We obtain the dot product of the sampled vectors on the unit sphere and our reciprocal peaks in source once
        unit_projections = unite_sphere_lattice @ source.flatten(0, 1).T
        unit_projections = unit_projections.unflatten(1, [bs, -1]).permute(1, 0, 2)

        # We use the information of the dot products with the unit
        # sphere to generate the dot products with the scaled sphere vectors
        projections = torch.stack([unit_projections * factor for factor in scaling], 0)
        h = torch.round(projections)

        # We explicitly create the candidates from the unit sphere mapping by triplicating it
        # and then scaling it accordingly
        unit_candidates = unite_sphere_lattice.expand(bs, 3, len(unite_sphere_lattice), 3)
        unit_candidates = unit_candidates * scaling[None, :, None, None]
        unit_candidates = unit_candidates.flatten(1, 2)

        diff = torch.abs(projections - h)
        is_inlier = diff <= dist_to_integer
        is_inlier &= (torch.sum(torch.abs(source), dim=-1) != 0)[None, :, None, :]

        combined_loss = torch.sum(is_inlier, dim=-1)

        indices = combined_loss.int().sort(descending=True, dim=-1).indices[:, :, 0: 50 * num_top_solutions].to(
            device)

        expanded_candidates = unit_candidates.unflatten(1, [3, -1]).permute(1, 0, 2, 3).flatten(0, 1)
        all_candidates = batched_subset_from_indices(expanded_candidates, indices.flatten(0, 1)).unflatten(0, [3, -1])

        # We perform a Trimmed Least Squares method with error threshold annealing to find the best vectors
        for d in [0.1, 0.05, 0.025, 0.01]:
            projections = torch.bmm(all_candidates.permute(1, 0, 2, 3).flatten(1, 2), source.permute(0, 2, 1))
            projections = projections.unflatten(1, [3, -1]).permute(1, 0, 2, 3)

            h = torch.round(projections)

            mask = torch.abs(h - projections) < d

            flat_h = h.permute(1, 0, 2, 3).reshape(bs * 3 * all_candidates.shape[2], -1).unsqueeze(-1)
            flat_mask = mask.permute(1, 0, 2, 3).reshape(bs * 3 * all_candidates.shape[2], -1)
            e_source = source.expand(3, all_candidates.shape[2], bs, -1, 3).permute(2, 0, 1, 3, 4).reshape(
                bs * 3 * all_candidates.shape[2], -1, 3)

            # Use a fitting but with a mask that takes only into account points closer to their target
            refined_candidates = torch.linalg.lstsq(
                e_source * flat_mask[:, :, None], flat_h * flat_mask[:, :, None]
            )[0]  # solutions is flatten bs, 3, len(unite_lattice)
            all_candidates = refined_candidates.unflatten(0, [bs, 3, -1]).squeeze(-1).transpose(0, 1)

        # After TLS, we score anr rank the resulting vectors and take only the top num_top_solutions
        diff = torch.abs(projections - h)
        is_inlier = diff <= 0.01
        is_inlier &= (torch.sum(torch.abs(source), dim=-1) != 0)[None, :, None, :]
        combined_loss = torch.sum(is_inlier, dim=-1)
        indices = combined_loss.int().sort(descending=True, dim=-1).indices[:, :, 0:num_top_solutions].to(
            device)
        expanded_candidates = all_candidates.flatten(0, 1)
        all_candidates = batched_subset_from_indices(expanded_candidates, indices.flatten(0, 1)).unflatten(0, [3, -1])

        if self.debugging:
            self.candidates = all_candidates.clone().detach()

        # We attach a basis to each of the candidates and rotate it around the candidate vector to produce
        # candidate bases which are 3x3
        rotated_bases = []
        for i, candidates in enumerate(all_candidates):
            permutation = [i, (i + 1) % 3, (i + 2) % 3]
            inverse_permutation = [(i + i) % 3, (i + 1 + i) % 3, (i + 2 + i) % 3]
            current_cell = initial_cell[permutation, :]
            all_basis = current_cell.repeat(candidates.shape[0], candidates.shape[1], 1, 1).to(device)

            R = rotation_to_target(all_basis[:, :, 0].flatten(0, 1), candidates.flatten(0, 1))
            bases = torch.bmm(R, all_basis.flatten(0, 1).transpose(1, 2)).permute(0, 2, 1)
            # recover the size of the candidate
            bases = torch.stack(
                [candidates.flatten(0, 1),
                 bases[:, 1, :],
                 bases[:, 2, :]], 1
            )

            alpha = 2 * math.pi * torch.arange(angle_resolution) / angle_resolution
            R = rotations(candidates.flatten(0, 1), alpha.to(device), angle_resolution)

            bts = bases.transpose(1, 2).repeat(angle_resolution, 1, 1, 1).transpose(0, 1)
            M = torch.bmm(R.reshape(-1, 3, 3), bts.reshape(-1, 3, 3)).permute(0, 2, 1)
            rotated_bases.append(M.view(-1, 3, 3)[:, inverse_permutation, :].unflatten(0, [bs, -1]))
        rotated_bases = torch.cat(rotated_bases, 1)
        return rotated_bases

    def compute_scores_and_hkl(self, source, rotated_bases, dist_to_integer: float = 0.12):
        rearanged_bases = rotated_bases.permute(0, 2, 3, 1)
        projections = torch.stack([
            torch.bmm(source, rearanged_bases[:, 0, :, :]),
            torch.bmm(source, rearanged_bases[:, 1, :, :]),
            torch.bmm(source, rearanged_bases[:, 2, :, :])
        ], -1).permute(0, 2, 1, 3)

        hkl = torch.round(projections)
        distance = torch.norm(projections - hkl, dim=-1, p=2)
        is_inlier = distance <= dist_to_integer
        scores = torch.sum(is_inlier, dim=-1)

        return scores, hkl, is_inlier

    def bfilter_solution(self, base, source, precision: float, min_num_spots: int):
        """
        Decides if the pair base, inliers makes a valid solution or not according to their error
        @param base: bs x 3 x 3 with a, b and c as row vectors in primal space.
        @param source: bs x k x 3 with k < n the set of spots indexed by base projected on the Ewald sphere in reciprocal space.
        @param min_num_spots: minimum number of spots of a valid solution
        @param precision: The precision of a valid solution
        @return: Sucess flag for each of the batches of base, i.e., a boolean vector of size bs
        """

        non_zero_mask = torch.sum(source ** 2, dim=-1) != 0

        predictions = torch.bmm(source, base.transpose(1, 2))
        hkl = torch.round(predictions)
        back_points = torch.bmm(hkl, torch.inverse(base).transpose(1, 2))
        errors = (source - back_points).norm(dim=-1, p=2)
        mask = errors < precision

        mask &= non_zero_mask

        return torch.sum(mask, dim=-1) >= min_num_spots

    def forward(self, source, initial_cell, min_num_spots: int, angle_resolution: int, num_top_solutions: int):
        """
        Indexes a batched instance
        @param source: bacthed 3D Points on the Ewald sphere bs, num_points, 3
        @param initial_cell: The given lattice cell in primal space 3,3
        @param min_num_spots: the minimum number of spots that a valid solution most have
        @param angle_resolution: The number of samples used to rotate the given bases
        @param num_top_solutions: The number of candidate solutions to be considered by the algorithm
        @return: solution_successes (boolean vector of size bs x num_crystals),
        solution_bases (solution cells bs x num_crystals x 3 x 3),
        solution_masks (boolean mask of source indicating elements of each solution bs x num_crystals x num_points) ,
        solution_errors (float tensor with the error of the solutions bs x num_crystals) ,
        solution_penalization (float tensor with the penalization used in the solutions bs x num_crystals)
        """
        with torch.no_grad():
            # fixed hyperparameter
            valid_integer_projection_radius = float(0.2)

            bs = int(len(source))

            candidate_bases = self.sample_bases(
                source,
                self.unite_sphere_lattice.to(source.device),
                initial_cell,
                dist_to_integer=valid_integer_projection_radius,
                num_top_solutions=num_top_solutions,
                angle_resolution=angle_resolution
            )

            if self.debugging:
                self.raw_bases = candidate_bases.clone().detach()

            scores, _, is_inlier = self.compute_scores_and_hkl(
                source,
                candidate_bases,
                dist_to_integer=valid_integer_projection_radius
            )

            # We take the indices of the top num_bases scores
            indices = scores.int().argsort(descending=True, dim=-1)[:, 0:num_top_solutions]
            # We consider from now on only the bases correspondig to these top indices
            candidate_bases = batched_subset_from_indices(candidate_bases, indices)

            if self.debugging:
                self.filtered_bases = candidate_bases.clone().detach()

            # We update the is_inlier mask with the same indices
            is_inlier = batched_subset_from_indices(is_inlier, indices)
            non_zero_mask = torch.sum(torch.abs(source), dim=-1) != 0
            is_inlier &= non_zero_mask.repeat(is_inlier.shape[1], 1, 1).transpose(0, 1)

            tuned_sources, top_bases, source_mask, error, penalization = self.index_candidate_solutions(
                candidate_bases,
                source.repeat(num_top_solutions, 1, 1, 1).transpose(0, 1),
                is_inlier,
                initial_cell,
                num_iterations=self.num_iterations
            )

            if self.debugging:
                self.top_bases = top_bases.clone().detach()

            # We extract iteratively the best solutions and ignore the spots they contain for subsequent rounds
            # while there is still a solution that contains at lest the min_num_spots, in this way we find multicrystals

            solution_bases = []
            solution_masks = []
            solution_errors = []
            solution_penalization = []
            solution_sources = []

            # we take the solution with the maximum number of inlier spots, but penalized
            solution_instance = torch.argmax(torch.sum(source_mask, dim=-1).int() - penalization, dim=-1)

            while torch.max(
                    torch.sum(batched_subset_from_indices(source_mask, solution_instance.unsqueeze(1)).squeeze(1),
                              dim=-1)).int() > min_num_spots:
                # We now find the best solution
                solution_instance = torch.argmax(torch.sum(source_mask, dim=-1).int(), dim=-1)
                solution_bases.append(
                    batched_subset_from_indices(top_bases, solution_instance.unsqueeze(1)).squeeze(1))
                solution_masks.append(
                    batched_subset_from_indices(source_mask, solution_instance.unsqueeze(1)).squeeze(1))
                solution_sources.append(
                    batched_subset_from_indices(tuned_sources, solution_instance.unsqueeze(1)).squeeze(1))
                solution_errors.append(batched_subset_from_indices(error, solution_instance.unsqueeze(1)).squeeze(1))
                solution_penalization.append(
                    batched_subset_from_indices(penalization, solution_instance.unsqueeze(1)).squeeze(1))

                # we turn to zeros the locations of source_mask of the spots belonging to the stored solution
                source_mask *= (~solution_masks[-1])[:, None, :]
                solution_instance = torch.argmax(torch.sum(source_mask, dim=-1).int() - penalization, dim=-1)

            if not solution_bases:
                return torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)

            solution_bases = torch.stack(solution_bases, 1)
            solution_masks = torch.stack(solution_masks, 1)
            solution_errors = torch.stack(solution_errors, 1)
            solution_penalization = torch.stack(solution_penalization, 1)
            solution_sources = torch.stack(solution_sources, 1)

            source_for_filtering = source.repeat(int(solution_bases.shape[1]), 1, 1, 1).transpose(0, 1)

            solution_successes = self.bfilter_solution(
                solution_bases.flatten(0, 1),
                source_for_filtering.flatten(0, 1) * solution_masks.flatten(0, 1)[:, :, None],
                precision=self.filter_precision,
                min_num_spots=self.filter_min_num_spots
            ).unflatten(0, [bs, -1])

            self.solution_sources = solution_sources
            return solution_successes, solution_bases, solution_masks, torch.max(solution_errors,
                                                                                   dim=-1).values, solution_penalization

    def index_candidate_solutions(
            self,
            bases,
            source,
            source_mask,
            initial_cell,
            num_iterations: int
    ):
        """
        @param bases: candidate basis solutions
        @param source: Points on the Ewald sphere in reciprocal space to index
        @param source_mask: A 0-1-mask of the source points indicating which are active
        @param initial_cell: The given cell defining the reciprocal lattice
        @param num_iterations: The num of iterations to run the algorithm
        @return: The candidate bases for the indexing, the source_mask, the error and the penalization
        """
        start = 0.01
        base = 2 ** (math.log(self.error_precision * 100) / ((num_iterations - 1) * math.log(2)))
        error_bounds = [float(start * base ** i) for i in range(num_iterations)]  # min error is 0.002
        error = torch.zeros_like(source_mask)
        for round in range(num_iterations):
            # We turn to zeros the locations that should not be considered for fitting
            masked_source = source * source_mask[:, :, :, None]
            targets = torch.round(masked_source @ bases.transpose(2, 3))
            # The regression intrinsically ignores all zero vectors that have zero targets as their residual is zero
            transposed_bases= torch.linalg.lstsq(masked_source, targets)[0]
            bases = transposed_bases.transpose(2, 3)

            # we compute now the inverse of the targets in reciprocal space which will be compared against source
            predictions = source @ transposed_bases
            back_points = torch.round(predictions) @ bb_inverse(transposed_bases)

            non_zero_mask = torch.sum(torch.abs(source), dim=-1) != 0

            # we compute now the new error
            error = torch.norm(source - back_points, dim=-1)

            source_mask = error < error_bounds[round]
            source_mask &= non_zero_mask

        # we should discard solutions that look at crystals differing too much from the given solution
        penalization = bcompute_penalization(
            bases,
            initial_cell.repeat(int(len(bases)), 1, 1)
        )

        return source, bases, source_mask, error * source_mask, penalization
