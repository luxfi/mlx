// Copyright © 2024 Lux Partners Limited
// NTT (Number Theoretic Transform) for lattice cryptography

#pragma once

#include "array.h"
#include "device.h"
#include "utils.h"

namespace mlx::core::ntt {

/**
 * NTT Context for batched operations.
 * Holds precomputed twiddle factors and Barrett reduction constants.
 */
struct NTTContext {
  uint32_t N = 0;       // Transform size (power of 2)
  uint64_t Q = 0;       // Modulus (NTT-friendly prime)
  uint64_t root = 0;    // Primitive N-th root of unity
  uint64_t inv_root = 0;// Inverse of root
  uint64_t inv_N = 0;   // Multiplicative inverse of N mod Q
  uint64_t barrett_mu = 0; // Barrett reduction constant

  array twiddles;       // Forward twiddle factors
  array inv_twiddles;   // Inverse twiddle factors

  // Default constructor with empty arrays
  NTTContext() : twiddles(array({})), inv_twiddles(array({})) {}

  // Constructor with values
  NTTContext(uint32_t n, uint64_t q, uint64_t r, uint64_t ir, uint64_t iN, uint64_t mu,
             array tw, array itw)
      : N(n), Q(q), root(r), inv_root(ir), inv_N(iN), barrett_mu(mu),
        twiddles(std::move(tw)), inv_twiddles(std::move(itw)) {}
};

/** Check if NTT GPU acceleration is available */
bool gpu_available(StreamOrDevice s = {});

/** Get current backend name */
const char* backend_name(StreamOrDevice s = {});

/** Create NTT context with precomputed twiddle factors */
NTTContext create_context(uint32_t N, uint64_t Q, StreamOrDevice s = {});

/** Destroy NTT context and release resources */
void destroy_context(NTTContext& ctx);

/**
 * Forward NTT (Cooley-Tukey decimation-in-time).
 *
 * Transforms polynomial from coefficient form to evaluation form.
 * Input: array of shape [batch, N] with dtype uint64
 * Output: array of shape [batch, N] with dtype uint64
 */
array forward(
    const NTTContext& ctx,
    const array& a,
    StreamOrDevice s = {});

/**
 * Inverse NTT (Gentleman-Sande decimation-in-frequency).
 *
 * Transforms polynomial from evaluation form back to coefficient form.
 * Input: array of shape [batch, N] with dtype uint64
 * Output: array of shape [batch, N] with dtype uint64
 */
array inverse(
    const NTTContext& ctx,
    const array& a,
    StreamOrDevice s = {});

/**
 * Pointwise modular multiplication of two NTT-transformed polynomials.
 *
 * Input: two arrays of shape [batch, N] with dtype uint64
 * Output: array of shape [batch, N] with dtype uint64
 */
array pointwise_mul(
    const NTTContext& ctx,
    const array& a,
    const array& b,
    StreamOrDevice s = {});

/**
 * Polynomial multiplication via NTT convolution.
 *
 * Computes: forward(a) * forward(b) -> inverse -> result
 * Input: two arrays of shape [batch, N] with dtype uint64
 * Output: array of shape [batch, N] with dtype uint64
 */
array polymul(
    const NTTContext& ctx,
    const array& a,
    const array& b,
    StreamOrDevice s = {});

} // namespace mlx::core::ntt
