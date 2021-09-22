#pragma once
#include <torch/extension.h>

std::vector<at::Tensor> ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
