#ifndef FRACTAL_DIM_H
#define FRACTAL_DIM_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <omp.h>
#include "diffu.h"

namespace fractal_detail {
constexpr size_t no_index = std::numeric_limits<size_t>::max();

// Indices are particle identities, so distinct particles at the same position
// remain neighbours. The scoped node array frees all memory after each call.
template <typename TYPEN>
class NeighborTree {
    struct Node {
        size_t index, left, right, subtree_end;
        PrecisionSetting lower[3], upper[3];
    };
    const std::vector<TYPEN>& positions;
    std::vector<Node> nodes;
    std::vector<size_t> source_to_node;
    size_t root;

    static PrecisionSetting coordinate(const TYPEN& point, int axis) {
        return axis == 0 ? point.x : (axis == 1 ? point.y : point.z);
    }
    size_t build(std::vector<size_t>& indices, size_t begin, size_t end, int depth) {
        if (begin == end) return no_index;
        const size_t middle = begin + (end - begin) / 2;
        const int axis = depth % 3;
        std::nth_element(indices.begin() + begin, indices.begin() + middle,
                         indices.begin() + end, [&](size_t a, size_t b) {
            const auto ca = coordinate(positions[a], axis);
            const auto cb = coordinate(positions[b], axis);
            return ca == cb ? a < b : ca < cb;
        });
        const size_t node_id = nodes.size();
        const TYPEN& point = positions[indices[middle]];
        nodes.push_back({indices[middle], no_index, no_index, no_index,
                         {point.x, point.y, point.z}, {point.x, point.y, point.z}});
        source_to_node[indices[middle]] = node_id;
        const size_t left = build(indices, begin, middle, depth + 1);
        const size_t right = build(indices, middle + 1, end, depth + 1);
        nodes[node_id].left = left;
        nodes[node_id].right = right;
        // Nodes are appended in preorder: every subtree is a contiguous range.
        nodes[node_id].subtree_end = nodes.size();
        for (size_t child : {left, right}) {
            if (child == no_index) continue;
            for (int axis = 0; axis < 3; ++axis) {
                nodes[node_id].lower[axis] = std::min(nodes[node_id].lower[axis], nodes[child].lower[axis]);
                nodes[node_id].upper[axis] = std::max(nodes[node_id].upper[axis], nodes[child].upper[axis]);
            }
        }
        return node_id;
    }
    size_t count(size_t node_id, const TYPEN& target, PrecisionSetting radius2,
                 size_t excluded_index, int depth) const {
        if (node_id == no_index) return 0;
        const Node& node = nodes[node_id];
        const TYPEN& point = positions[node.index];
        const PrecisionSetting dx = point.x - target.x;
        const PrecisionSetting dy = point.y - target.y;
        const PrecisionSetting dz = point.z - target.z;
        size_t result = node.index != excluded_index && dx*dx + dy*dy + dz*dz <= radius2;
        if (node.left == no_index && node.right == no_index) return result;
        PrecisionSetting nearest2 = 0, farthest2 = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const PrecisionSetting value = coordinate(target, axis);
            const PrecisionSetting dlo = node.lower[axis] - value;
            const PrecisionSetting dhi = node.upper[axis] - value;
            if (dlo > 0) nearest2 += dlo*dlo;
            else if (dhi < 0) nearest2 += dhi*dhi;
            farthest2 += std::max(dlo*dlo, dhi*dhi);
        }
        if (nearest2 > radius2) return 0;
        if (farthest2 <= radius2) {
            const size_t excluded_node = excluded_index == no_index ? no_index : source_to_node[excluded_index];
            return node.subtree_end - node_id -
                (excluded_node >= node_id && excluded_node < node.subtree_end);
        }
        const PrecisionSetting delta = coordinate(target, depth % 3) - coordinate(point, depth % 3);
        if (delta*delta <= radius2) {
            result += count(node.left, target, radius2, excluded_index, depth + 1);
            result += count(node.right, target, radius2, excluded_index, depth + 1);
        } else {
            result += count(delta < 0 ? node.left : node.right, target, radius2,
                            excluded_index, depth + 1);
        }
        return result;
    }
public:
    explicit NeighborTree(const std::vector<TYPEN>& input) : positions(input) {
        nodes.reserve(input.size());
        source_to_node.resize(input.size());
        std::vector<size_t> indices(input.size());
        std::iota(indices.begin(), indices.end(), 0);
        root = build(indices, 0, indices.size(), 0);
    }
    size_t count(const TYPEN& target, PrecisionSetting radius, size_t excluded_index) const {
        return count(root, target, radius * radius, excluded_index, 0);
    }
};
}

/** Mean cumulative number of neighbours with distance <= r.
 * target_indices gives each target's index in positions; precisely that particle
 * is excluded, including when other particles coincide with it. With no IDs,
 * positions itself as targets implies IDs 0..N-1; other targets are external
 * query centres and all source particles count. Copied subsets must supply IDs.
 * This measures cumulative counts, not shell counts or a fitted dimension.
 */
template <typename TYPEN>
std::pair<std::vector<PrecisionSetting>, std::vector<PrecisionSetting>>
computeFractalDimensionNeighbors(
    const std::vector<TYPEN>& positions, const std::vector<TYPEN>& targets,
    PrecisionSetting r_min, PrecisionSetting r_max, int n_radii,
    const std::vector<size_t>& target_indices = {}
) {
    if (n_radii <= 0 || positions.empty() || targets.empty()) return {};
    if (!std::isfinite(r_min) || !std::isfinite(r_max) || r_min < 0 || r_max < r_min ||
        (n_radii > 1 && r_min == 0))
        throw std::invalid_argument("Neighbour radii require 0 < r_min <= r_max (zero allowed for one radius).");
    if (!target_indices.empty() && target_indices.size() != targets.size())
        throw std::invalid_argument("Each target requires one source index.");
    for (size_t i = 0; i < target_indices.size(); ++i) {
        const size_t index = target_indices[i];
        if (index == fractal_detail::no_index) continue; // explicitly external centre
        if (index >= positions.size()) throw std::out_of_range("Target source index is out of range.");
        if (positions[index].x != targets[i].x || positions[index].y != targets[i].y ||
            positions[index].z != targets[i].z)
            throw std::invalid_argument("Target coordinates do not match its source index.");
    }
    std::vector<PrecisionSetting> radii(n_radii, r_min), avg_counts(n_radii);
    if (n_radii > 1) {
        const PrecisionSetting dlog = (std::log(r_max) - std::log(r_min)) / (n_radii - 1);
        for (int i = 1; i < n_radii; ++i) radii[i] = std::exp(std::log(r_min) + dlog*i);
        radii.back() = r_max;
    }
    const fractal_detail::NeighborTree<TYPEN> tree(positions);
    std::vector<unsigned long long> total(n_radii, 0);
    const bool all_source_targets = &positions == &targets;
#pragma omp parallel
    {
        std::vector<unsigned long long> local_total(n_radii, 0);
#pragma omp for schedule(static)
        for (size_t i = 0; i < targets.size(); ++i) {
            const size_t excluded = !target_indices.empty() ? target_indices[i] :
                (all_source_targets ? i : fractal_detail::no_index);
            for (int ri = 0; ri < n_radii; ++ri)
                local_total[ri] += tree.count(targets[i], radii[ri], excluded);
        }
#pragma omp critical(fractal_count_sum)
        for (int ri = 0; ri < n_radii; ++ri) total[ri] += local_total[ri];
    }
    for (int ri = 0; ri < n_radii; ++ri)
        avg_counts[ri] = static_cast<PrecisionSetting>(total[ri]) / targets.size();
    return {radii, avg_counts};
}

inline std::pair<PrecisionSetting, PrecisionSetting> fractal_count_bounds(
    size_t N, PrecisionSetting scale, PrecisionSetting lower=-1.,
    PrecisionSetting upper=-1., int n_radii=12
) {
    if (!N || !std::isfinite(scale) || scale<=0 || n_radii<1 ||
        !std::isfinite(lower) || !std::isfinite(upper))
        throw std::invalid_argument("Invalid neighbour-count scale, size or bounds.");
    const auto lo=lower>=0?lower:scale*std::pow(9.*std::acos(-1.)/(2.*N),1./3.)*.5;
    const auto hi=upper>=0?upper:scale*.25;
    if (lo<0 || hi<lo || (n_radii>1 && (lo==0 || hi==lo)))
        throw std::invalid_argument("Neighbour counting needs 0 < lower < upper for multiple radii; supply explicit bounds for small N.");
    return {lo,hi};
}

/** Save cumulative counts separately from fitting. Existing four-argument calls
 * retain their historical 12 radii: scale*(9*pi/(2*N))^(1/3)/2 to scale/4.
 * target_count > 0 overrides rate_calculate. seed controls the sampled IDs.
 * Optional count bounds affect the saved diagnostic coverage, not a fit window.
 */
template <typename TYPEN>
void calculate_frac_dim(
    const std::vector<TYPEN>& positions, PrecisionSetting scale, std::string file_save,
    double rate_calculate = 1.0, uint64_t seed = 5489u, size_t target_count = 0,
    PrecisionSetting count_rmin = -1., PrecisionSetting count_rmax = -1., int n_radii = 12
) {
    if (positions.empty() || !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("Fractal counting requires a nonempty sample and positive scale.");
    if (n_radii < 1)
        throw std::invalid_argument("n_radii must be positive.");
    const size_t N = positions.size();
    const auto bounds=fractal_count_bounds(N,scale,count_rmin,count_rmax,n_radii);
    const PrecisionSetting r_min=bounds.first, r_max=bounds.second;
    if (!target_count && (!std::isfinite(rate_calculate) || rate_calculate <= 0 || rate_calculate > 1)) {
        // Legacy callers use 1e4/N, which exceeds one in their small-N checks.
        std::cerr << "rate_calculate outside (0, 1]; using all source particles.\n";
        rate_calculate = 1.;
    }
    const size_t n_targets = target_count ? std::min(target_count, N) :
        std::max(size_t(1), static_cast<size_t>(rate_calculate * N));
    std::vector<size_t> ids(N);
    std::iota(ids.begin(), ids.end(), 0);
    if (n_targets < N) {
        std::mt19937_64 generator(seed);
        std::shuffle(ids.begin(), ids.end(), generator);
        ids.resize(n_targets);
    }
    std::vector<TYPEN> targets;
    targets.reserve(ids.size());
    for (size_t id : ids) targets.push_back(positions[id]);
    std::cout << "Cumulative neighbour count: r_min=" << r_min << " r_max=" << r_max
              << " n_radii=" << n_radii << " n_targets=" << n_targets << " seed=" << seed << '\n';
    const double start_time = omp_get_wtime();
    const auto result = computeFractalDimensionNeighbors(positions, targets, r_min, r_max, n_radii, ids);
    std::cout << "Neighbour counting took " << omp_get_wtime() - start_time << " seconds.\n";
    std::ofstream output(file_save);
    if (!output)
        throw std::runtime_error("Failed to save cumulative neighbour counts: " + file_save);
    output << std::setprecision(17)
           << "# statistic: mean cumulative neighbour count; distance <= radius\n"
           << "# self_exclusion: original source index; distinct coincident particles count\n"
           << "# N=" << N << " n_targets=" << n_targets << " seed=" << seed
           << " scale=" << scale << " n_radii=" << n_radii << '\n'
           << "# radius  mean_N_le_radius\n";
    for (size_t i = 0; i < result.first.size(); ++i)
        output << result.first[i] << ' ' << result.second[i] << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("Failed while writing cumulative neighbour counts: " + file_save);
}
#endif // FRACTAL_DIM_H
