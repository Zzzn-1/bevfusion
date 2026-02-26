#include <torch/torch.h>

// CUDA function declarations
// void feature_decorator(int64_t64_t b, int64_t64_t d, int64_t64_t h, int64_t64_t w, int64_t64_t n, int64_t64_t c, int64_t64_t n_int64_t64_tervals, const float* x,
//     const int64_t64_t* geom_feats, const int64_t64_t* int64_t64_terval_starts, const int64_t64_t* int64_terval_lengths, float* out);

void feature_decorator(float* out);

at::Tensor feature_decorator_forward(
  const at::Tensor _x, 
  const at::Tensor _y, 
  const at::Tensor _z, 
  const double vx, const double vy, const double x_offset, const double y_offset, 
  int64_t normalize_coords, int64_t use_cluster, int64_t use_center
) {
  int64_t n = _x.size(0);
  int64_t c = _x.size(1);
  int64_t a = _x.size(2);
  auto options = torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  int64_t decorate_dims = 0;
  if (use_cluster > 0) {
    decorate_dims += 3;
  }
  if (use_center > 0) {
    decorate_dims += 2;
  }

  at::Tensor _out = torch::zeros({n, c, a+decorate_dims}, options);
  float* out = _out.data_ptr<float>();
  const float* x = _x.data_ptr<float>();
  const int64_t* y = _y.data_ptr<int64_t>();
  const int64_t* z = _z.data_ptr<int64_t>();
  feature_decorator(out);
  return _out;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("feature_decorator_forward", &feature_decorator_forward,
        "feature_decorator_forward");
}

static auto registry =
    torch::RegisterOperators("feature_decorator_ext::feature_decorator_forward", &feature_decorator_forward);
