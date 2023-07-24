import torch
import torch.nn as nn
import math
from numpy import arange, pi, sin, cos, arccos
from torch.nn.functional import normalize
import sys


def batched_invert_matrix(A):
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


def compute_center(source):
    """
    Estimate the center of a bunch of points sampled on the boundary of the sphere
    @param source: n,3 points on the surface of the sphere
    @return: The center and radius of the sphere
    """
    lift = torch.stack([
        2 * source[:, 0],
        2 * source[:, 1],
        2 * source[:, 2],
        torch.ones(len(source)).to(source.device)
    ], 1).to(source.device)

    targets = source[:, 0] ** 2 + source[:, 1] ** 2 + source[:, 2] ** 2

    s, _, _, _ = torch.linalg.lstsq(lift, targets.unsqueeze(1))
    s = s.reshape(-1).to(source.device)
    return s[:3], torch.sqrt(s[3] + s[0] ** 2 + s[1] ** 2 + s[2] ** 2)


def compute_intersection_with_sphere(source, center):
    normalized_source = normalize(source, dim=-1)

    # Calculate the projection of the centers onto the rays in the directions of normalized_source
    t = torch.sum(normalized_source * center, dim=-1)

    # Since the closest point lies exactly in the middle between the center and the desired inetrsection
    intersections = 2 * t[:, :, :, None] * normalized_source

    return intersections


def rotations(rodrigues_vector, alpha, angle_resolution: int):
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


def batched_compute_angles(triples):
    triples = torch.nn.functional.normalize(triples, dim=-1)
    alpha = torch.acos(torch.sum(triples[:, 0] * triples[:, 1], dim=-1)) * 180 / math.pi
    beta = torch.acos(torch.sum(triples[:, 0] * triples[:, 2], dim=-1)) * 180 / math.pi
    gamma = torch.acos(torch.sum(triples[:, 1] * triples[:, 2], dim=-1)) * 180 / math.pi
    return torch.stack([alpha, beta, gamma], 1)


def compute_penalization(matrix, initial_cell):
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
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, num_points)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / num_points)
    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);
    lattice = torch.randn(num_points, 3)
    lattice[:, 0] = torch.FloatTensor(x)
    lattice[:, 1] = torch.FloatTensor(y)
    lattice[:, 2] = torch.FloatTensor(z)
    return lattice


def unsqueeze_if_needed(s, len_shape_single_item: int):
    if len(s.shape) == len_shape_single_item:
        return s.unsqueeze(0), True
    elif len(s.shape) == len_shape_single_item + 1:
        return s, False
    else:
        raise Exception("Not a valid number of dimension for s")


def to_skew_symmetric(x):  # bs, 3
    x, unsqueezed = unsqueeze_if_needed(x, len_shape_single_item=1)
    out = torch.zeros(x.shape[0], 3, 3).to(x.device)
    out[:, 0, 1] = x[:, 0]
    out[:, 0, 2] = x[:, 1]
    out[:, 1, 0] = -x[:, 0]
    out[:, 1, 2] = x[:, 2]
    out[:, 2, 0] = -x[:, 1]
    out[:, 2, 1] = -x[:, 2]
    return out.squeeze(0) if unsqueezed else out


def rotation_to_target(sources, targets):
    sources, _ = unsqueeze_if_needed(sources, len_shape_single_item=1)
    targets, unsqueezed = unsqueeze_if_needed(targets, len_shape_single_item=1)

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
    return R if not unsqueezed else R.squeeze(0)


class IndexerModule(nn.Module):
    def __init__(self, lattice_size, num_iterations: int = 10, error_precision: float = 0.0018,
                 filter_precision: float = 0.0012, filter_min_num_spots: int = 6,
                 ewald_correction: bool = True, ewald_thickness: float = 0.005):
        super(IndexerModule, self).__init__()
        self.unite_sphere_lattice = torch.nn.Parameter(create_sphere_lattice(num_points=lattice_size))
        self.filter_precision = float(filter_precision)
        self.filter_min_num_spots = int(filter_min_num_spots)
        gettrace = getattr(sys, 'gettrace', None)
        self.debugging = gettrace is not None
        # self.debugging = False
        self.ewald_correction = bool(ewald_correction)
        self.error_precision = float(error_precision)
        self.ewald_thickness = float(ewald_thickness)
        self.num_iterations = int(num_iterations)
        self.solution_sources = torch.zeros(0)

    def bcompute_penalization(self, X, Y):
        first = int(X.shape[0])
        second = int(X.shape[1])
        result = compute_penalization(X.flatten(0, 1), Y.flatten(0, 1))
        return result.unflatten(0, [first, second])

    def bb_inverse(self, X):
        first = int(X.shape[0])
        second = int(X.shape[1])
        result = batched_invert_matrix(X.flatten(0, 1))
        return result.unflatten(0, [first, second])

    def compute_triples(
            self,
            source,
            unite_sphere_lattice,
            initial_cell: torch.Tensor,
            dist_to_integer: float = 0.2,
            num_top_solutions: int = 1000,
            angle_resolution: int = 360
    ):
        """
        Computes the triple systems that are going to be candidates for solutions.
        Takes the first vector from the unite_sphere_lattice scaled by the length of the first vefctor of the initial_cell,
        and filters possible solutions by looking at those that have more integer projections.
        Then it attaches to each of those a copy of the initial_cell, and rotataes them angle_resolution times by keeping the first vector fixed.
        In this way it obtains a set of candidate triple systems
        @param source: The spots as 3D points projected on the Ewald sphere
        @param unite_sphere_lattice: A sample of points on the unit sphere
        @param initial_cell: The given initial basis cell for this molecule
        @param dist_to_integer: The maximum allowed distance for a projection to its closest integer to be considered as a true spot
        @param num_top_solutions: the number of vectors from the scaled unite_sphere_lattice that will be considered for the second part of the algorithm
        @param angle_resolution: The number of initial_cell that will be attached to each of the candidate vectors in the first direction.
        @return: A tensor of the form kx3x3, containing the top k triple systems, where k = num_top_solutions
        """

        device = source.device
        bs = int(source.shape[0])
        scaling = initial_cell.norm(dim=-1, p=2).to(device)

        # We obtain the dot product of the sampled vectors on the unit sphere and our reciprocal peaks in source once
        unit_projections = unite_sphere_lattice @ source.flatten(0, 1).T
        unit_projections = unit_projections.unflatten(1, [bs, -1]).permute(1, 0, 2)

        # We use the information of the dot products with the unit sphere to generate the dot products with the scaled sphere vectors
        projections = torch.stack([unit_projections * factor for factor in scaling], 0)
        h = torch.round(projections)

        # Using only the sampling of the sphere as candidates
        candidates = unite_sphere_lattice.expand(bs, 3, len(unite_sphere_lattice), 3)
        candidates = candidates.clone() * scaling[None, :, None, None]
        candidates = candidates.flatten(1, 2)

        projections = torch.bmm(candidates, source.permute(0, 2, 1))
        projections = projections.unflatten(1, [3, -1]).permute(1, 0, 2, 3)

        h = torch.round(projections)

        diff = torch.abs(projections - h)
        is_inlier = diff <= dist_to_integer
        is_inlier &= (torch.sum(torch.abs(source), dim=-1) != 0)[None, :, None, :]

        combined_loss = torch.sum(is_inlier, dim=-1)

        indices = combined_loss.int().sort(descending=True, dim=-1).indices[:, :, 0: 50 * num_top_solutions].to(
            device)

        expanded_candidates = candidates.unflatten(1, [3, -1]).permute(1, 0, 2, 3).flatten(0, 1)
        all_candidates = batched_subset_from_indices(expanded_candidates, indices.flatten(0, 1)).unflatten(0, [3, -1])

        # We do fitting of the directions
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
            refined_candidates = torch.linalg.lstsq(e_source * flat_mask[:, :, None], flat_h * flat_mask[:, :, None])[
                0]  # solutions is flatten bs, 3, len(unite_lattice)
            all_candidates = refined_candidates.unflatten(0, [bs, 3, -1]).squeeze(-1).transpose(0, 1)

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

        rotated_triple_systems = []
        for i, candidates in enumerate(all_candidates):
            permutation = [i, (i + 1) % 3, (i + 2) % 3]
            inverse_permutation = [(i + i) % 3, (i + 1 + i) % 3, (i + 2 + i) % 3]
            current_cell = initial_cell[permutation, :]
            all_basis = current_cell.repeat(candidates.shape[0], candidates.shape[1], 1, 1).to(device)

            R = rotation_to_target(all_basis[:, :, 0].flatten(0, 1), candidates.flatten(0, 1))
            triple_systems = torch.bmm(R, all_basis.flatten(0, 1).transpose(1, 2)).permute(0, 2, 1)
            # recover the size of the candidate
            triple_systems = torch.stack(
                [candidates.flatten(0, 1),
                 triple_systems[:, 1, :],
                 triple_systems[:, 2, :]], 1
            )

            alpha = 2 * math.pi * torch.arange(angle_resolution) / angle_resolution
            R = rotations(candidates.flatten(0, 1), alpha.to(device), angle_resolution)

            bts = triple_systems.transpose(1, 2).repeat(angle_resolution, 1, 1, 1).transpose(0, 1)
            M = torch.bmm(R.reshape(-1, 3, 3), bts.reshape(-1, 3, 3)).permute(0, 2, 1)
            rotated_triple_systems.append(M.view(-1, 3, 3)[:, inverse_permutation, :].unflatten(0, [bs, -1]))
        rotated_triple_systems = torch.cat(rotated_triple_systems, 1)
        return rotated_triple_systems

    def compute_scores_and_hkl(self, source, rotated_triple_systems, dist_to_integer: float = 0.12):
        rearanged_triple_systems = rotated_triple_systems.permute(0, 2, 3, 1)
        projections = torch.stack([
            torch.bmm(source, rearanged_triple_systems[:, 0, :, :]),
            torch.bmm(source, rearanged_triple_systems[:, 1, :, :]),
            torch.bmm(source, rearanged_triple_systems[:, 2, :, :])
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
        @param angle_resolution: The number of samples used to rotate the given triples
        @param num_top_solutions: The number of candidate solutions to be considered by the algorithm
        @return: solution_successes (boolean vector of size bs x num_crystals),
        solution_triples (solution cells bs x num_crystals x 3 x 3),
        solution_masks (boolean mask of source indicating elements of each solution bs x num_crystals x num_points) ,
        solution_errors (float tensor with the error of the solutions bs x num_crystals) ,
        solution_penalization (float tensor with the penalization used in the solutions bs x num_crystals)
        """
        with torch.no_grad():
            # crank up parameters if num spots is low
            if source.shape[1] <= 30:
                num_top_solutions = int(500)
                angle_resolution = int(500)
            elif source.shape[1] <= 50:
                num_top_solutions *= 2
                angle_resolution *= 2

            # paramters
            valid_integer_projection_radius = float(0.2)

            bs = int(len(source))

            rotated_triple_systems = self.compute_triples(
                source,
                self.unite_sphere_lattice.to(source.device),
                initial_cell,
                dist_to_integer=valid_integer_projection_radius,
                num_top_solutions=int(num_top_solutions // 2),
                angle_resolution=angle_resolution
            )

            if self.debugging:
                self.raw_triples = rotated_triple_systems.clone().detach()

            scores, _, is_inlier = self.compute_scores_and_hkl(
                source,
                rotated_triple_systems,
                dist_to_integer=valid_integer_projection_radius
            )

            # We take the indices of the top num_triples scores
            indices = scores.int().argsort(descending=True, dim=-1)[:, 0:num_top_solutions]
            # We consider from now on only the triples correspondig to these top indices
            rotated_triple_systems = batched_subset_from_indices(rotated_triple_systems, indices)

            if self.debugging:
                self.filtered_triples = rotated_triple_systems.clone().detach()

            # We update the is_inlier mask with the same indices
            is_inlier = batched_subset_from_indices(is_inlier, indices)
            non_zero_mask = torch.sum(torch.abs(source), dim=-1) != 0
            is_inlier &= non_zero_mask.repeat(is_inlier.shape[1], 1, 1).transpose(0, 1)

            tuned_sources, top_triples, source_mask, error, penalization = self.index_candidate_solutions(
                rotated_triple_systems,
                source.repeat(num_top_solutions, 1, 1, 1).transpose(0, 1),
                is_inlier,
                initial_cell,
                num_iterations=self.num_iterations
            )

            if self.debugging:
                self.top_triples = top_triples.clone().detach()

            # We extract iteratively the best solutions and ignore the spots they contain for subsequent rounds
            # while there is still a solution that contains at lest the min_num_spots, in this way we fin dmulti crystals

            solution_triples = []
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
                solution_triples.append(
                    batched_subset_from_indices(top_triples, solution_instance.unsqueeze(1)).squeeze(1))
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

            if not solution_triples:
                return torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)

            solution_triples = torch.stack(solution_triples, 1)
            solution_masks = torch.stack(solution_masks, 1)
            solution_errors = torch.stack(solution_errors, 1)
            solution_penalization = torch.stack(solution_penalization, 1)
            solution_sources = torch.stack(solution_sources, 1)

            source_for_filtering = source.repeat(int(solution_triples.shape[1]), 1, 1, 1).transpose(0, 1)

            solution_successes = self.bfilter_solution(
                solution_triples.flatten(0, 1),
                source_for_filtering.flatten(0, 1) * solution_masks.flatten(0, 1)[:, :, None],
                precision=self.filter_precision,
                min_num_spots=self.filter_min_num_spots
            ).unflatten(0, [bs, -1])

            self.solution_sources = solution_sources
            return solution_successes, solution_triples, solution_masks, torch.max(solution_errors,
                                                                                   dim=-1).values, solution_penalization

    def index_candidate_solutions(
            self,
            triples,
            source,
            source_mask,
            initial_cell,
            num_iterations: int
    ):
        """
        @param triples: candidate triple solutions
        @param source: Points on the Ewald sphere in reciprocal space to index
        @param source_mask: A 0-1-mask of the source points indicating which are active
        @param initial_cell: The given cell defining the reciprocal lattice
        @param num_iterations: The num of iterations to run the algorithm
        @return: The candidate triples for the indexing, the source_mask, the error and the penalization
        """
        start = 0.01
        base = 2 ** (math.log(self.error_precision * 100) / ((num_iterations - 1) * math.log(2)))
        error_bounds = [float(start * base ** i) for i in range(num_iterations)]  # min error is 0.002
        error = torch.zeros_like(source_mask)
        for round in range(num_iterations):
            # We turn to zeros the locations that should not be considered for fitting
            masked_source = source * source_mask[:, :, :, None]
            targets = torch.round(masked_source @ triples.transpose(2, 3))
            # The regression intrinsically ignores all zero vectors that have zero targets as their residual is zero
            transposed_triples = torch.linalg.lstsq(masked_source, targets)[0]
            triples = transposed_triples.transpose(2, 3)

            # we compute now the inverse of the targets in reciprocal space which will be compared against source
            predictions = source @ transposed_triples
            back_points = torch.round(predictions) @ self.bb_inverse(transposed_triples)

            non_zero_mask = torch.sum(torch.abs(source), dim=-1) != 0

            # Guesses the position of the reciprocal peaks in source so that it matches best the targets
            # This uses the freedom of the width of the Ewald sphere and is thought for pink-beam data
            if self.ewald_correction:
                # estimate the wavelength
                center, radius = compute_center(source[0][0])
                wavelength = 1 / radius
                ewald_thickness = self.ewald_thickness

                # each point in source is allowed to move in the portion of the ray going from the origin
                # to the reciprocal point that lies inside the width of the Ewald sphere.

                small_radius = float(1 / wavelength * (1 + ewald_thickness))
                large_radius = float(1 / wavelength * (1 - ewald_thickness))

                small_center = small_radius * center / radius
                large_center = large_radius * center / radius

                # We compute the endpoints of the valid segment for each point in source
                source_small = compute_intersection_with_sphere(source, small_center)
                source_large = compute_intersection_with_sphere(source, large_center)

                # We compute the closest point in the valid segment to the target point and move the
                # reciprocal points to these locations.
                A = source_large[non_zero_mask] - source_small[non_zero_mask]
                B = back_points[non_zero_mask] - source_small[non_zero_mask]

                normalized_A = normalize(A, dim=-1)

                # Calculate the projection of the centers onto the rays in the directions of normalized_source
                t = torch.sum(normalized_A * B, dim=-1)
                t = torch.clip(t, torch.zeros_like(t), A.norm(dim=-1, p=2))

                # Since the closest point lies exactly in the middle between the center and the desired inetrsection
                new_source = source_small[non_zero_mask] + t[:, None] * normalized_A

                source = torch.zeros_like(source)
                source[non_zero_mask] = new_source

            # we compute now the new error
            error = torch.norm(source - back_points, dim=-1)

            source_mask = error < error_bounds[round]
            source_mask &= non_zero_mask

        # we should discard solutions that look at crystals differing too much from the given solution
        penalization = self.bcompute_penalization(
            triples,
            initial_cell.repeat(int(len(triples)), 1, 1)
        )

        return source, triples, source_mask, error * source_mask, penalization
