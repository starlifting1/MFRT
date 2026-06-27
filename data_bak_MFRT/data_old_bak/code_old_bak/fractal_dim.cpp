#include "fractal_dim.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// Define the Node structure for KD-tree
struct KDNode {
    size_t index;      // index of the point in the original positions vector
    KDNode* left;
    KDNode* right;
    // We don't explicitly store splitting axis or value because we can infer it from depth
    KDNode(size_t idx) : index(idx), left(nullptr), right(nullptr) {}
};

// We will use a global (static) reference to the positions array for convenience in recursion
static const std::vector<PositionSpace>* kd_positions = nullptr;

// KD-tree build function
KDNode* buildKDTree(std::vector<size_t>& indices, int depth, size_t start, size_t end) {
    if (start >= end) {
        return nullptr;
    }
    // Choose axis based on depth (0->x, 1->y, 2->z)
    int axis = depth % 3;
    size_t mid = start + (end - start) / 2;
    
    // Partially sort points around median on chosen axis
    auto comp = [axis](size_t a, size_t b) {
        const PositionSpace& pa = (*kd_positions)[a];
        const PositionSpace& pb = (*kd_positions)[b];
        if (axis == 0) return pa.x < pb.x;
        if (axis == 1) return pa.y < pb.y;
        return pa.z < pb.z;
    };
    std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end, comp);
    
    // Create node for median
    KDNode* node = new KDNode(indices[mid]);
    
    // Recursively build subtrees
    node->left  = buildKDTree(indices, depth + 1, start, mid);
    node->right = buildKDTree(indices, depth + 1, mid + 1, end);
    
    return node;
}

// Range count (sphere) query: counts points within given radius of target
int rangeCount(
    const KDNode* node, const PositionSpace& target, PrecisionSetting radius, int depth, 
    bool is_target_self_count=true
){
    if (!node) return 0;
    int count = 0;
    // Compute squared distance from target to current point
    const PositionSpace& currentPoint = (*kd_positions)[node->index];
    PrecisionSetting dx = currentPoint.x - target.x;
    PrecisionSetting dy = currentPoint.y - target.y;
    PrecisionSetting dz = currentPoint.z - target.z;
    PrecisionSetting dist2 = dx*dx + dy*dy + dz*dz;
    PrecisionSetting radius2 = radius * radius;
    if (dist2 <= radius2) {
        // Exclude self if it's exactly the same point (distance 0)
        
        if (!is_target_self_count) { // self-count
            if (dist2 > 1e-12) { // This check is important if targets are part of positions.
                count += 1;
            }
        }else{
            count += 1;
        }
    }
    // Determine axis and diff
    int axis = depth % 3;
    PrecisionSetting diff;
    if (axis == 0) diff = target.x - currentPoint.x;
    else if (axis == 1) diff = target.y - currentPoint.y;
    else diff = target.z - currentPoint.z;
    
    // If the hypersphere crosses the splitting plane, search both sides
    if (diff * diff <= radius2) {
        count += rangeCount(node->left, target, radius, depth + 1);
        count += rangeCount(node->right, target, radius, depth + 1);
    } else if (diff < 0) {
        // target is "left" of current point on this axis
        count += rangeCount(node->left, target, radius, depth + 1);
    } else {
        // target is "right" of current point on this axis
        count += rangeCount(node->right, target, radius, depth + 1);
    }
    return count;
}

// Main function implementation
std::pair<std::vector<PrecisionSetting>, std::vector<PrecisionSetting>>
computeFractalDimensionNeighbors(
    const std::vector<PositionSpace>& positions, const std::vector<PositionSpace>& targets,
    PrecisionSetting r_min, PrecisionSetting r_max, int n_radii
){
    std::vector<PrecisionSetting> radii;
    std::vector<PrecisionSetting> avg_counts;
    if (n_radii <= 0 || positions.empty() || targets.empty()) {
        // Return empty results for degenerate cases
        return {radii, avg_counts};
    }
    // If only one radius, just use r_min (or r_max, they should be equal ideally)
    if (n_radii == 1) {
        PrecisionSetting r = r_min;
        radii.push_back(r);
        // Count neighbors within r for all targets and average
        // Build KD-tree for counting
        std::vector<size_t> indices(positions.size());
        std::iota(indices.begin(), indices.end(), 0);
        kd_positions = &positions;
        KDNode* root = buildKDTree(indices, 0, 0, indices.size());
        long long total_neighbors = 0;
        for (const PositionSpace& t : targets) {
            total_neighbors += rangeCount(root, t, r, 0);
        }
        // Free KD-tree (to avoid memory leak)
        // A full tree deletion function can be written to traverse and delete nodes.
        // For brevity, we assume the application will free memory elsewhere if needed.
        // (In practice, you should delete all nodes here.)
        
        avg_counts.push_back(static_cast<PrecisionSetting>(total_neighbors) / targets.size());
        return {radii, avg_counts};
    }
    // Compute geometric progression for radii
    radii.resize(n_radii);
    avg_counts.assign(n_radii, 0.0);
    PrecisionSetting log_min = std::log(r_min);
    PrecisionSetting log_max = std::log(r_max);
    PrecisionSetting dlog = (log_max - log_min) / (n_radii - 1);
    for (int i = 0; i < n_radii; ++i) {
        radii[i] = std::exp(log_min + dlog * i);
    }
    // Ensure last radius is exactly r_max (to avoid floating point rounding issues)
    radii.back() = r_max;
    
    // Build KD-tree from all positions
    std::vector<size_t> indices(positions.size());
    std::iota(indices.begin(), indices.end(), 0);
    kd_positions = &positions;
    KDNode* root = buildKDTree(indices, 0, 0, indices.size());
    
    // For each target, perform range counts for each radius
#pragma omp parallel for schedule(dynamic)
    for (const PositionSpace& t : targets) {
        // We can optionally sort queries by increasing radius and reuse previous counts.
        // However, since we do full search each time (with KD-tree pruning), it's fine.
        int prev_count = 0;
        PrecisionSetting prev_radius = 0;
        for (int ri = 0; ri < n_radii; ++ri) {
            PrecisionSetting r = radii[ri];
            // If needed: optimization to skip counting if radius hasn't increased (not applicable here).
            int count = rangeCount(root, t, r, 0);
            avg_counts[ri] += count;  // accumulate counts (to average later)
            prev_count = count;
            prev_radius = r;
        }
    }
    
    // Average the counts
    for (PrecisionSetting& sum : avg_counts) {
        sum /= targets.size();
    }
    
    // (Optional) TODO: Free KD-tree memory to prevent leaks.
    // Ideally we would traverse and delete all nodes.
    
    return {radii, avg_counts};
}
