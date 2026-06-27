#ifndef FRACTAL_DIM_H
#define FRACTAL_DIM_H

#include <vector>
#include <utility>  // for std::pair
#include <omp.h>
#include "diffu.h"

/**
 * Computes the geometric sequence of radii and the average number of neighbors within each radius.
 * 
 * @param positions List of particle positions in space.
 * @param targets List of target center positions to count neighbors around.
 * @param r_min Minimum radius (inclusive).
 * @param r_max Maximum radius (inclusive).
 * @param n_radii Number of radii (points in geometric sequence between r_min and r_max).
 * @return A pair of vectors: 
 *         first: radii (size n_radii), 
 *         second: average neighbor counts for each radius (size n_radii).
 */
std::pair<std::vector<PrecisionSetting>, std::vector<PrecisionSetting>>
computeFractalDimensionNeighbors(
    const std::vector<PositionSpace>& positions, const std::vector<PositionSpace>& targets,
    PrecisionSetting r_min, PrecisionSetting r_max, int n_radii
);

#endif // FRACTAL_DIM_H
