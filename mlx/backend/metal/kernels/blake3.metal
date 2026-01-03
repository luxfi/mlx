// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BLAKE3 Cryptographic Hash - High-performance parallel hashing
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

constant uint IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

constant uint MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}
};

constant uint CHUNK_START = 1u;
constant uint CHUNK_END = 2u;
constant uint PARENT = 4u;
constant uint ROOT = 8u;

inline uint rotr(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

inline void g(thread uint* state, uint a, uint b, uint c, uint d, uint mx, uint my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr(state[d] ^ state[a], 16u);
    state[c] = state[c] + state[d];
    state[b] = rotr(state[b] ^ state[c], 12u);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr(state[d] ^ state[a], 8u);
    state[c] = state[c] + state[d];
    state[b] = rotr(state[b] ^ state[c], 7u);
}

inline void round_fn(thread uint* state, thread uint* msg, uint round_idx) {
    uint s = round_idx % 7;
    g(state, 0, 4, 8, 12, msg[MSG_SCHEDULE[s][0]], msg[MSG_SCHEDULE[s][1]]);
    g(state, 1, 5, 9, 13, msg[MSG_SCHEDULE[s][2]], msg[MSG_SCHEDULE[s][3]]);
    g(state, 2, 6, 10, 14, msg[MSG_SCHEDULE[s][4]], msg[MSG_SCHEDULE[s][5]]);
    g(state, 3, 7, 11, 15, msg[MSG_SCHEDULE[s][6]], msg[MSG_SCHEDULE[s][7]]);
    g(state, 0, 5, 10, 15, msg[MSG_SCHEDULE[s][8]], msg[MSG_SCHEDULE[s][9]]);
    g(state, 1, 6, 11, 12, msg[MSG_SCHEDULE[s][10]], msg[MSG_SCHEDULE[s][11]]);
    g(state, 2, 7, 8, 13, msg[MSG_SCHEDULE[s][12]], msg[MSG_SCHEDULE[s][13]]);
    g(state, 3, 4, 9, 14, msg[MSG_SCHEDULE[s][14]], msg[MSG_SCHEDULE[s][15]]);
}

inline void compress(thread uint* cv, thread uint* block, uint counter, uint block_len, uint flags, thread uint* out) {
    uint state[16];
    state[0] = cv[0]; state[1] = cv[1]; state[2] = cv[2]; state[3] = cv[3];
    state[4] = cv[4]; state[5] = cv[5]; state[6] = cv[6]; state[7] = cv[7];
    state[8] = IV[0]; state[9] = IV[1]; state[10] = IV[2]; state[11] = IV[3];
    state[12] = counter; state[13] = 0u; state[14] = block_len; state[15] = flags;
    
    for (uint r = 0; r < 7; r++) {
        round_fn(state, block, r);
    }
    
    for (uint i = 0; i < 8; i++) {
        out[i] = state[i] ^ state[i + 8];
        out[i + 8] = state[i + 8] ^ cv[i];
    }
}

kernel void blake3_hash_chunks(
    device const uint* input [[buffer(0)]],
    device uint* chunk_cvs [[buffer(1)]],
    constant uint& num_chunks [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_chunks) return;
    
    uint cv[8];
    for (uint i = 0; i < 8; i++) cv[i] = IV[i];
    
    uint chunk_offset = gid * 256; // 1024 bytes = 256 uint
    
    for (uint block_idx = 0; block_idx < 16; block_idx++) {
        uint block[16];
        for (uint i = 0; i < 16; i++) {
            block[i] = input[chunk_offset + block_idx * 16 + i];
        }
        
        uint flags = 0u;
        if (block_idx == 0) flags |= CHUNK_START;
        if (block_idx == 15) flags |= CHUNK_END;
        
        uint out[16];
        compress(cv, block, gid, 64u, flags, out);
        
        for (uint i = 0; i < 8; i++) cv[i] = out[i];
    }
    
    uint cv_offset = gid * 8;
    for (uint i = 0; i < 8; i++) {
        chunk_cvs[cv_offset + i] = cv[i];
    }
}

kernel void blake3_merge_parents(
    device uint* chunk_cvs [[buffer(0)]],
    constant uint& num_parents [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_parents) return;
    
    uint left_offset = gid * 2 * 8;
    uint right_offset = left_offset + 8;
    
    uint block[16];
    for (uint i = 0; i < 8; i++) {
        block[i] = chunk_cvs[left_offset + i];
        block[i + 8] = chunk_cvs[right_offset + i];
    }
    
    uint cv[8];
    for (uint i = 0; i < 8; i++) cv[i] = IV[i];
    
    uint flags = PARENT;
    if (num_parents == 1) flags |= ROOT;
    
    uint out[16];
    compress(cv, block, 0u, 64u, flags, out);
    
    uint out_offset = gid * 8;
    for (uint i = 0; i < 8; i++) {
        chunk_cvs[out_offset + i] = out[i];
    }
}

kernel void blake3_hash_small(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& input_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    uint cv[8];
    for (uint i = 0; i < 8; i++) cv[i] = IV[i];
    
    uint num_blocks = (input_len + 63) / 64;
    
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint block[16];
        for (uint i = 0; i < 16; i++) {
            uint byte_offset = block_idx * 64 + i * 4;
            block[i] = (byte_offset < input_len) ? input[block_idx * 16 + i] : 0u;
        }
        
        uint remaining = input_len - block_idx * 64;
        uint block_len = min(remaining, 64u);
        
        uint flags = 0u;
        if (block_idx == 0) flags |= CHUNK_START;
        if (block_idx == num_blocks - 1) flags |= CHUNK_END | ROOT;
        
        uint out[16];
        compress(cv, block, 0u, block_len, flags, out);
        
        if (block_idx == num_blocks - 1) {
            for (uint i = 0; i < 8; i++) output[i] = out[i];
        } else {
            for (uint i = 0; i < 8; i++) cv[i] = out[i];
        }
    }
}
