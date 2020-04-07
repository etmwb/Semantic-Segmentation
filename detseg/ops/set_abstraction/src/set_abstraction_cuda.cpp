// modify from
// https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib/

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

void gather_points_cuda_forward(int b, int c, int n, int npoints,
                                at::Tensor data_pt, at::Tensor data_idx,
                                at::Tensor data_out);

void gather_points_cuda_backward(int b, int c, int n, int npoints,
                                 at::Tensor grad_out, at::Tensor data_idx,
                                 at::Tensor grad_pt);

void furthest_point_sampling_cuda(int b, int n, int m,
                                  at::Tensor data_pt, at::Tensor data_tmp,
                                  at::Tensor data_idx);

void three_nn_cuda(int b, int n, int m, at::Tensor data_unk,
                   at::Tensor data_kno, at::Tensor data_dis, at::Tensor data_idx);

void three_interpolate_cuda_forward(int b, int c, int m, int n,
                                    at::Tensor data_pt, at::Tensor data_idx,
                                    at::Tensor data_wt, at::Tensor data_out);

void three_interpolate_cuda_backward(int b, int c, int n, int m,
                                     at::Tensor grad_out,
                                     at::Tensor data_idx, at::Tensor data_wt,
                                     at::Tensor grad_pt);

void group_points_cuda_forward(int b, int c, int n, int npoints, int nsamples,
                               at::Tensor data_pt, at::Tensor data_idx,
                               at::Tensor data_out);

void group_points_cuda_backward(int b, int c, int n, int npoints,
                                int nsamples, at::Tensor grad_out,
                                at::Tensor data_idx, at::Tensor grad_pt);

void ball_query_cuda(int b, int n, int m, float radius,
                     int nsamples, at::Tensor data_nxyz,
                     at::Tensor data_xyz, at::Tensor data_idx);


at::Tensor gather_points_forward(at::Tensor points, at::Tensor idxs) {
    AT_CHECK(points.is_contiguous(), "points tensor has to be contiguous");
    AT_CHECK(idxs.is_contiguous(), "idxs tensor has to be contiguous");

    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idxs.size(1)},
                      points.options());

    gather_points_cuda_forward(points.size(0), points.size(1), points.size(2),
                                 idxs.size(1), points, idxs, output);

    return output;
}

at::Tensor gather_points_backward(at::Tensor grad_out, at::Tensor idxs,
                              const int n) {
    AT_CHECK(grad_out.is_contiguous(), "grad_out tensor has to be contiguous");
    AT_CHECK(idxs.is_contiguous(), "idxs tensor has to be contiguous");

    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), n},
                      grad_out.options());

    gather_points_cuda_backward(grad_out.size(0), grad_out.size(1), n,
                                idxs.size(1), grad_out, idxs, output);

    return output;
}

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
    AT_CHECK(points.is_contiguous(), "points tensor has to be contiguous");

    at::Tensor idxs =
        torch::zeros({points.size(0), nsamples},
                      at::device(points.device()).dtype(at::ScalarType::Int));
    at::Tensor tmp =
        torch::full({points.size(0), points.size(1)}, 1e10,
                     points.options());

    furthest_point_sampling_cuda(
        points.size(0), points.size(1), nsamples, points, tmp, idxs);

    return idxs;
}

std::vector<at::Tensor> three_nn(at::Tensor unknown, at::Tensor known) {
    AT_CHECK(unknown.is_contiguous(), "unknown tensor has to be contiguous"); 
    AT_CHECK(known.is_contiguous(), "known tensor has to be contiguous"); 

    at::Tensor idxs =
        torch::zeros({unknown.size(0), unknown.size(1), 3},
                      at::device(unknown.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 =
        torch::zeros({unknown.size(0), unknown.size(1), 3},
                      unknown.options());

    three_nn_cuda(unknown.size(0), unknown.size(1), known.size(1),
                  unknown, known, dist2, idxs);

    return {dist2, idxs};
}

at::Tensor three_interpolate_forward(at::Tensor points, at::Tensor idxs,
                                     at::Tensor weight) { // points means features in previous layer.
    AT_CHECK(points.is_contiguous(), "points tensor has to be contiguous"); 
    AT_CHECK(idxs.is_contiguous(), "idxs tensor has to be contiguous"); 
    AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous"); 

    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idxs.size(1)},
                      points.options());

    three_interpolate_cuda_forward(
        points.size(0), points.size(1), points.size(2), idxs.size(1),
        points, idxs, weight, output);
  
    return output;
}

at::Tensor three_interpolate_backward(at::Tensor grad_out, at::Tensor idxs,
                                      at::Tensor weight, const int m) {
    AT_CHECK(grad_out.is_contiguous(), "grad_out tensor has to be contiguous"); 
    AT_CHECK(idxs.is_contiguous(), "idxs tensor has to be contiguous"); 
    AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous"); 

    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), m},
                      grad_out.options());

    three_interpolate_cuda_backward(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out, idxs, weight, output);

    return output;
}

at::Tensor group_points_forward(at::Tensor points, at::Tensor idxs) {
    AT_CHECK(points.is_contiguous(), "points tensor has to be contiguous"); 
    AT_CHECK(idxs.is_contiguous(), "idxs tensor has to be contiguous"); 

    at::Tensor output =
        torch::zeros({points.size(0), points.size(1), idxs.size(1), idxs.size(2)},
                      points.options());

    group_points_cuda_forward(points.size(0), points.size(1), points.size(2),
                              idxs.size(1), idxs.size(2), points, idxs, output);

    return output;
}

at::Tensor group_points_backward(at::Tensor grad_out, at::Tensor idxs, const int n) {
    AT_CHECK(grad_out.is_contiguous(), "grad_out tensor has to be contiguous"); 
    AT_CHECK(idxs.is_contiguous(), "idxs tensor has to be contiguous"); 

    at::Tensor output =
        torch::zeros({grad_out.size(0), grad_out.size(1), n},
                      grad_out.options());

    group_points_cuda_backward(
        grad_out.size(0), grad_out.size(1), n, idxs.size(1), idxs.size(2),
        grad_out, idxs, output);
  
    return output;
}

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsamples) { 
    AT_CHECK(new_xyz.is_contiguous(), "new_xyz tensor has to be contiguous");
    AT_CHECK(xyz.is_contiguous(), "xyz tensor has to be contiguous");

    at::Tensor idxs =
        torch::zeros({new_xyz.size(0), new_xyz.size(1), nsamples},
                      at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    ball_query_cuda(xyz.size(0), xyz.size(1), new_xyz.size(1),
                    radius, nsamples, new_xyz, xyz, idxs);

    return idxs
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sampling", &furthest_point_sampling,
        "furthest point sampling (CUDA)");
  m.def("gather_points_forward", &gather_points_forward,
        "gather points forward (CUDA)");
  m.def("gather_points_backward", &gather_points_backward,
        "gather points backward (CUDA)");
  m.def("three_nn", &three_nn,
        "three nn (CUDA)");
  m.def("three_interpolate_forward", &three_interpolate_forward,
        "three interpolate forward (CUDA)");
  m.def("three_interpolate_backward", &three_interpolate_backward,
        "three interpolate backward (CUDA)");
  m.def("group_points_forward", &group_points_forward,
        "group points forward (CUDA)");
  m.def("group_points_backward", &group_points_backward,
        "group points backward (CUDA)");
  m.def("ball_query", &ball_query,
        "ball query (CUDA)");
}