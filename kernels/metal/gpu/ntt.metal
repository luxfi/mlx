// Copyright © 2024 Lux Partners Limited
// Metal NTT kernel instantiation for lattice cryptography

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/ntt.h"

// Instantiate fused forward NTT kernels for common sizes
#define instantiate_ntt_forward(N) \
  instantiate_kernel("ntt_forward_fused_" #N, ntt_forward_fused, N)

#define instantiate_ntt_inverse(N) \
  instantiate_kernel("ntt_inverse_fused_" #N, ntt_inverse_fused, N)

// Common NTT sizes for lattice cryptography:
// - 256: Kyber/Dilithium
// - 512: Extended Kyber
// - 1024: Large Dilithium
// - 2048: General lattice ops
// - 4096: Max size that fits in 32KB threadgroup memory (8 bytes * 4096 = 32KB)

instantiate_ntt_forward(256)
instantiate_ntt_forward(512)
instantiate_ntt_forward(1024)
instantiate_ntt_forward(2048)
instantiate_ntt_forward(4096)

instantiate_ntt_inverse(256)
instantiate_ntt_inverse(512)
instantiate_ntt_inverse(1024)
instantiate_ntt_inverse(2048)
instantiate_ntt_inverse(4096)

// Non-templated kernels are instantiated directly from the header:
// - ntt_pointwise_mul
// - ntt_forward_stage
// - ntt_inverse_stage
// - ntt_scale
