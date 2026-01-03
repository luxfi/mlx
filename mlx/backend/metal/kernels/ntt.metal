// Copyright © 2024 Lux Partners Limited
// Metal NTT kernel instantiations

#include "mlx/backend/metal/kernels/ntt.h"

// Instantiate fused NTT kernels for common sizes
#define instantiate_ntt_fused(N)                                        \
  template [[host_name("ntt_forward_fused_" #N)]]                       \
  [[kernel]] void ntt_forward_fused<N>(                                 \
      device ulong* data [[buffer(0)]],                                 \
      constant const ulong* twiddles [[buffer(1)]],                     \
      constant const ulong& Q [[buffer(2)]],                            \
      constant const ulong& mu [[buffer(3)]],                           \
      constant const uint& batch [[buffer(4)]],                         \
      uint3 tid [[thread_position_in_threadgroup]],                     \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 tgSize [[threads_per_threadgroup]]);                        \
                                                                        \
  template [[host_name("ntt_inverse_fused_" #N)]]                       \
  [[kernel]] void ntt_inverse_fused<N>(                                 \
      device ulong* data [[buffer(0)]],                                 \
      constant const ulong* inv_twiddles [[buffer(1)]],                 \
      constant const ulong& Q [[buffer(2)]],                            \
      constant const ulong& mu [[buffer(3)]],                           \
      constant const ulong& inv_N [[buffer(4)]],                        \
      constant const uint& batch [[buffer(5)]],                         \
      uint3 tid [[thread_position_in_threadgroup]],                     \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 tgSize [[threads_per_threadgroup]]);

// Instantiate for common NTT sizes
instantiate_ntt_fused(256)
instantiate_ntt_fused(512)
instantiate_ntt_fused(1024)
instantiate_ntt_fused(2048)
instantiate_ntt_fused(4096)
