#include <Eigen/Dense>
#include <cstdint>
#include <tuple>

std::tuple<Eigen::ArrayX2d, Eigen::Array<int64_t, Eigen::Dynamic, 1>> collect_inputs_trunk_layers_preliminary_fitting(
    Eigen::ArrayX3d trunk_layer_xyz, Eigen::Array<int64_t, Eigen::Dynamic, 1> cluster_labels,
    Eigen::Array<int64_t, Eigen::Dynamic, 1> unique_cluster_labels, double trunk_search_min_z,
    int64_t trunk_search_circle_fitting_num_layers, double trunk_search_circle_fitting_layer_height,
    double trunk_search_circle_fitting_layer_overlap, int64_t trunk_search_circle_fitting_min_points = 0,
    int num_workers = -1);

std::tuple<Eigen::ArrayX2d, Eigen::Array<int64_t, Eigen::Dynamic, 1>, Eigen::ArrayXd>
collect_inputs_trunk_layers_exact_fitting(Eigen::ArrayX3d trunk_layer_xyz,
                                          Eigen::Array<double, Eigen::Dynamic, 5> preliminary_layer_circles_or_ellipses,
                                          double trunk_search_min_z, int64_t trunk_search_circle_fitting_num_layers,
                                          double trunk_search_circle_fitting_layer_height,
                                          double trunk_search_circle_fitting_layer_overlap,
                                          double trunk_search_circle_fitting_switch_buffer_threshold,
                                          double trunk_search_circle_fitting_small_buffer_width,
                                          double trunk_search_circle_fitting_large_buffer_width,
                                          int64_t trunk_search_circle_fitting_min_points = 0, int num_workers = -1);

Eigen::Array<int64_t, Eigen::Dynamic, 1> segment_tree_crowns(
    Eigen::ArrayX3d xyz, Eigen::ArrayXd distance_to_dtm, Eigen::Array<bool, Eigen::Dynamic, 1> is_tree,
    Eigen::ArrayX2d tree_positions, Eigen::ArrayXd trunk_diameters, double region_growing_voxel_size,
    double region_growing_z_scale, double region_growing_seed_layer_height, double region_growing_seed_radius_factor,
    double region_growing_min_total_assignment_ratio, double region_growing_min_tree_assignment_ratio,
    double region_growing_max_search_radius, int64_t region_growing_decrease_search_radius_after_num_iter,
    int64_t region_growing_max_iterations, double region_growing_cum_search_dist_include_terrain, int num_workers = -1);
