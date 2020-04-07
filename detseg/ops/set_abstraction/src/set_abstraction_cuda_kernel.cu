#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS 512

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads =
        max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);
    return block_config;
}

template <typename scalar_t>
__global__ void gather_points_cuda_forward_gpu_kernel(int b, int c, int n, int m,
                                                      const scalar_t *__restrict__ points,
                                                      const int *__restrict__ idx,
                                                      scalar_t *__restrict__ out) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) { // remove first two loop
            for (int j = threadIdx.x; j < m; j += blockDim.x) {
                int a = idx[i * m + j];
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

void gather_points_cuda_forward(int b, int c, int n, int npoints,
                                at::Tensor data_pt, at::Tensor data_idx,
                                at::Tensor data_out) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_pt.scalar_type(), "gather_points_cuda_forward", ([&] {
        const scalar_t *data_pt_ = data_pt.data<scalar_t>();
        scalar_t *data_out_ = data_out.data<scalar_t>();
        const int *data_idx_ = data_idx.data<int>();

        gather_points_cuda_forward_gpu_kernel
            <<<dim3(b, c, 1), opt_n_threads(npoints)>>>(b, c, n, npoints, data_pt_, data_idx_, data_out_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in gather_points_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void gather_points_cuda_backward_gpu_kernel(int b, int c, int n, int m,
                                                       const scalar_t *__restrict__ grad_out,
                                                       const int *__restrict__ idx,
                                                       scalar_t *__restrict__ grad_points) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) { // remove first two loop
            for (int j = threadIdx.x; j < m; j += blockDim.x) {
                int a = idx[i * m + j];
                atomicAdd(grad_points + (i * c + l) * n + a, 
                          grad_out[(i * c + l) * m + j]);
            }
        }
    }
}

void gather_points_cuda_backward(int b, int c, int n, int npoints,
                                 at::Tensor grad_out, at::Tensor data_idx,
                                 at::Tensor grad_pt) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_out.scalar_type(), "gather_points_cuda_backward", ([&] {
        const scalar_t *grad_out_ = grad_out.data<scalar_t>();
        scalar_t *grad_pt_ = grad_pt.data<scalar_t>();
        const int *data_idx_ = data_idx.data<int>();

        gather_points_cuda_backward_gpu_kernel
            <<<dim3(b, c, 1), opt_n_threads(npoints)>>>(b, c, n, npoints, grad_out_, data_idx_, grad_pt_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in gather_points_cuda_backward: %s\n", cudaGetErrorString(err));
    }
}


template <typename scalar_t>
__device__ void __update(scalar_t *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) { 
    // avoid thread divergence, More details can be found in 
    // Programming Massively Parallel Processors A Hands-On Approach(Sec 5.3)
    const scalar_t v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <typename scalar_t>
template <unsigned int block_size>
__global__ void furthest_point_sampling_gpu_kernel(
    int b, int n, int m, const scalar_t *__restrict__ dataset,
    scalar_t *__restrict__ temp, int *__restrict__ idxs) {
    if (m <= 0) return;
    __shared__ scalar_t dists[block_size];
    __shared__ int dists_i[block_size];
    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;
    int tid = threadIdx.x;
    const int stride = block_size;
    int old = 0; // last determined idx
    if (threadIdx.x == 0) idxs[0] = old;
    __syncthreads();
    for (int j = 1; j < m; j++) { // iteration process
        int besti = 0;
        scalar_t best = -1;
        scalar_t x1 = dataset[old * 3 + 0];
        scalar_t y1 = dataset[old * 3 + 1];
        scalar_t z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride) {
            scalar_t x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            if (mag <= 1e-3) continue;
            scalar_t d =
                (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            scalar_t d2 = min(d, temp[k]);
            temp[k] = d2; //distance between current point k and all previous determined points 
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();
        if (block_size >= 512) {
          if (tid < 256) {
            __update<scalar_t>(dists, dists_i, tid, tid + 256);
          }
          __syncthreads();
        }
        if (block_size >= 256) {
          if (tid < 128) {
            __update<scalar_t>(dists, dists_i, tid, tid + 128);
          }
          __syncthreads();
        }
        if (block_size >= 128) {
          if (tid < 64) {
            __update<scalar_t>(dists, dists_i, tid, tid + 64);
          }
          __syncthreads();
        }
        if (block_size >= 64) {
          if (tid < 32) {
            __update<scalar_t>(dists, dists_i, tid, tid + 32);
          }
          __syncthreads();
        }
        if (block_size >= 32) {
          if (tid < 16) {
            __update<scalar_t>(dists, dists_i, tid, tid + 16);
          }
          __syncthreads();
        }
        if (block_size >= 16) {
          if (tid < 8) {
            __update<scalar_t>(dists, dists_i, tid, tid + 8);
          }
          __syncthreads();
        }
        if (block_size >= 8) {
          if (tid < 4) {
            __update<scalar_t>(dists, dists_i, tid, tid + 4);
          }
          __syncthreads();
        }
        if (block_size >= 4) {
          if (tid < 2) {
            __update<scalar_t>(dists, dists_i, tid, tid + 2);
          }
          __syncthreads();
        }
        if (block_size >= 2) {
          if (tid < 1) {
            __update<scalar_t>(dists, dists_i, tid, tid + 1);
          }
          __syncthreads();
        }
        old = dists_i[0];
        if (tid == 0) idxs[j] = old;
    }
}

void furthest_point_sampling_cuda(int b, int n, int m,
                                  at::Tensor data_pt, at::Tensor data_tmp,
                                  at::Tensor data_idx)
{
    unsigned int n_threads = opt_n_threads(n);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        points.scalar_type(), "furthest_point_sampling_gpu", ([&] {
        const scalar_t *data_pt_ = data_pt.data<scalar_t>();
        scalar_t *data_tmp_ = data_tmp.data<scalar_t>();
        int *data_idx_ = data_idx.data<int>();

        switch (n_threads) {
            case 512: 
                furthest_point_sampling_gpu_kernel<512>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 256: 
                furthest_point_sampling_gpu_kernel<256>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 128: 
                furthest_point_sampling_gpu_kernel<128>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 64: 
                furthest_point_sampling_gpu_kernel<64>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 32: 
                furthest_point_sampling_gpu_kernel<32>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 16: 
                furthest_point_sampling_gpu_kernel<16>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 8: 
                furthest_point_sampling_gpu_kernel<8>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 4: 
                furthest_point_sampling_gpu_kernel<4>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 2: 
                furthest_point_sampling_gpu_kernel<2>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            case 1: 
                furthest_point_sampling_gpu_kernel<1>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
            default: 
                furthest_point_sampling_gpu_kernel<512>
                    <<<b, n_threads>>>(b, n, m, data_pt_, data_tmp_, data_idx_);
                break;
        }        
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in furthest_point_sampling_cuda: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void three_nn_gpu_kernel(int b, int n, int m,
                                    const scalar_t *__restrict__ unknown,
                                    const scalar_t *__restrict__ known,
                                    scalar_t *__restrict__ dist2,
                                    int *__restrict__ idx) {
    int batch_index = blockIdx.x;
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * 3;
    idx += batch_index * n * 3;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride) {
        scalar_t ux = unknown[j * 3 + 0];
        scalar_t uy = unknown[j * 3 + 1];
        scalar_t uz = unknown[j * 3 + 2];

        scalar_t best1 = 1e40, best2 = 1e40, best3 = 1e40;
        int besti1 = 0, besti2 = 0, besti3 = 0;
        for (int k = 0; k < m; ++k) {
            scalar_t x = known[k * 3 + 0];
            scalar_t y = known[k * 3 + 1];
            scalar_t z = known[k * 3 + 2];
            scalar_t d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            if (d < best1) {
                best3 = best2;
                besti3 = besti2;
                best2 = best1;
                besti2 = besti1;
                best1 = d;
                besti1 = k;
            } else if (d < best2) {
                best3 = best2;
                besti3 = besti2;
                best2 = d;
                besti2 = k;
            } else if (d < best3) {
                best3 = d;
                besti3 = k;
            }
        }
        dist2[j * 3 + 0] = best1;
        dist2[j * 3 + 1] = best2;
        dist2[j * 3 + 2] = best3;

        idx[j * 3 + 0] = besti1;
        idx[j * 3 + 1] = besti2;
        idx[j * 3 + 2] = besti3;
    }
}

void three_nn_cuda(int b, int n, int m, at::Tensor data_unk,
                   at::Tensor data_kno, at::Tensor data_dis, at::Tensor data_idx) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_unk.scalar_type(), "three_nn_cuda", ([&] {
        const scalar_t *data_unk_ = data_unk.data<scalar_t>();
        const scalar_t *data_kno_ = data_kno.data<scalar_t>();
        scalar_t *data_dis_ = data_dis.data<scalar_t>();
        int *data_idx_ = data_idx.data<int>();

        three_nn_gpu_kernel
            <<<b, opt_n_threads(n)>>>(b, n, m, data_unk_, data_kno_, data_dis_, data_idx_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in three_nn_cuda: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void three_interpolate_cuda_forward_gpu_kernel(int b, int c, int m, int n,
                                                          const scalar_t *__restrict__ points,
                                                          const int *__restrict__ idx,
                                                          const scalar_t *__restrict__ weight,
                                                          scalar_t *__restrict__ out) {
    int batch_index = blockIdx.x;
    points += batch_index * m * c;

    idx += batch_index * n * 3;
    weight += batch_index * n * 3;

    out += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * n; i += stride) {
        const int l = i / n;
        const int j = i % n;
        scalar_t w1 = weight[j * 3 + 0];
        scalar_t w2 = weight[j * 3 + 1];
        scalar_t w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 +
                 points[l * m + i3] * w3;
    }
}

void three_interpolate_cuda_forward(int b, int c, int m, int n,
                                    at::Tensor data_pt, at::Tensor data_idx,
                                    at::Tensor data_wt, at::Tensor data_out) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_unk.scalar_type(), "three_interpolate_cuda_forward", ([&] {
        const scalar_t *data_pt_ = data_pt.data<scalar_t>();
        const scalar_t *data_wt_ = data_wt.data<scalar_t>();
        scalar_t *data_out_ = data_out.data<scalar_t>();
        const int *data_idx_ = data_idx.data<int>();

        three_interpolate_cuda_forward_gpu_kernel
            <<<b, opt_block_config(n, c)>>>(b, c, m, n, data_pt_, data_idx_, data_wt_, data_out_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in three_interpolate_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void three_interpolate_cuda_backward_gpu_kernel(
    int b, int c, int n, int m, const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx, const scalar_t *__restrict__ weight,
    scalar_t *__restrict__ grad_points) {
    int batch_index = blockIdx.x;
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * n; i += stride) {
        const int l = i / n;
        const int j = i % n;
        scalar_t w1 = weight[j * 3 + 0];
        scalar_t w2 = weight[j * 3 + 1];
        scalar_t w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        atomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
        atomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
        atomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
    }
}

void three_interpolate_cuda_backward(int b, int c, int n, int m,
                                     at::Tensor grad_out,
                                     at::Tensor data_idx, at::Tensor data_wt,
                                     at::Tensor grad_pt) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_out.scalar_type(), "three_interpolate_cuda_backward", ([&] {
        const scalar_t *grad_out_ = grad_out.data<scalar_t>();
        const scalar_t *data_wt_ = data_wt.data<scalar_t>();
        scalar_t *grad_pt_ = grad_pt.data<scalar_t>();
        const int *data_idx_ = data_idx.data<int>();

        three_interpolate_cuda_backward_gpu_kernel
            <<<b, opt_block_config(n, c)>>>(b, c, n, m, grad_out_, data_idx_, grad_wt_, grad_pt_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in three_interpolate_cuda_backward: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void group_points_cuda_forward_gpu_kernel(int b, int c, int n, int npoints,
                                                     int nsample,
                                                     const scalar_t *__restrict__ points,
                                                     const int *__restrict__ idx,
                                                     scalar_t *__restrict__ out) {
    int batch_index = blockIdx.x;
    points += batch_index * n * c;
    idx += batch_index * npoints * nsample;
    out += batch_index * npoints * nsample * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * npoints; i += stride) {
        const int l = i / npoints;
        const int j = i % npoints;
        for (int k = 0; k < nsample; ++k) {
            int ii = idx[j * nsample + k];
            out[(l * npoints + j) * nsample + k] = points[l * n + ii];
        }
    }
}

void group_points_cuda_forward(int b, int c, int n, int npoints, int nsamples,
                               at::Tensor data_pt, at::Tensor data_idx,
                               at::Tensor data_out) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_pt.scalar_type(), "group_points_cuda_forward", ([&] {
        const scalar_t *data_pt_ = data_pt.data<scalar_t>();
        scalar_t *data_out_ = data_out.data<scalar_t>();
        const int *data_idx_ = data_idx.data<int>();

        group_points_cuda_forward_gpu_kernel
            <<<b, opt_block_config(npoints, c)>>>(b, c, n, npoints, nsamples, data_pt_, data_idx_, data_out_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in group_points_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void group_points_cuda_backward_gpu_kernel(int b, int c, int n, int npoints,
                                                      int nsample,
                                                      const scalar_t *__restrict__ grad_out,
                                                      const int *__restrict__ idx,
                                                      scalar_t *__restrict__ grad_points) {
    int batch_index = blockIdx.x;
    grad_out += batch_index * npoints * nsample * c;
    idx += batch_index * npoints * nsample;
    grad_points += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index; i < c * npoints; i += stride) {
        const int l = i / npoints;
        const int j = i % npoints;
        for (int k = 0; k < nsample; ++k) {
            int ii = idx[j * nsample + k];
            atomicAdd(grad_points + l * n + ii,
                      grad_out[(l * npoints + j) * nsample + k]);
        }
    }
}

void group_points_cuda_backward(int b, int c, int n, int npoints,
                                int nsamples, at::Tensor grad_out,
                                at::Tensor data_idx, at::Tensor grad_pt) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_out.scalar_type(), "group_points_cuda_backward", ([&] {
        const scalar_t *grad_out_ = grad_out.data<scalar_t>();
        scalar_t *grad_pt_ = grad_pt.data<scalar_t>();
        const int *data_idx_ = data_idx.data<int>();

        group_points_cuda_backward_gpu_kernel
            <<<b, opt_block_config(npoints, c)>>>(b, c, n, npoints, nsamples, grad_out_, data_idx_, grad_pt_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in group_points_cuda_backward: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
__global__ void ball_query_cuda_gpu_kernel(int b, int n, int m, float radius,
                                           int nsample,
                                           const scalar_t *__restrict__ new_xyz,
                                           const scalar_t *__restrict__ xyz,
                                           int *__restrict__ idx) {
    int batch_index = blockIdx.x;
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    idx += m * nsample * batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    scalar_t radius2 = radius * radius;
    for (int j = index; j < m; j += stride) {
        scalar_t new_x = new_xyz[j * 3 + 0];
        scalar_t new_y = new_xyz[j * 3 + 1];
        scalar_t new_z = new_xyz[j * 3 + 2];
        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
            scalar_t x = xyz[k * 3 + 0];
            scalar_t y = xyz[k * 3 + 1];
            scalar_t z = xyz[k * 3 + 2];
            scalar_t d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                       (new_z - z) * (new_z - z);
            if (d2 < radius2) {
                if (cnt == 0) {
                    for (int l = 0; l < nsample; ++l) {
                        idx[j * nsample + l] = k;
                    }
                }
                idx[j * nsample + cnt] = k;
                ++cnt;
            }
        }
    }
}

void ball_query_cuda(int b, int n, int m, float radius,
                     int nsamples, at::Tensor data_nxyz,
                     at::Tensor data_xyz, at::Tensor data_idx) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_xyz.scalar_type(), "ball_query_cuda", ([&] {
        const scalar_t *data_nxyz_ = data_nxyz.data<scalar_t>();
        const scalar_t *data_xyz_ = data_xyz.data<scalar_t>();
        int *data_idx_ = data_idx.data<int>();

        ball_query_cuda_gpu_kernel
            <<<b, opt_n_threads(m)>>>(b, n, m, radius, nsamples, data_nxyz_, data_xyz_, data_idx_);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ball_query_cuda: %s\n", cudaGetErrorString(err));
    }
}