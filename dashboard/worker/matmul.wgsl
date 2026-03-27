// VRAMancer WebGPU Worker — Tiled Matrix Multiplication Shader
//
// Compute C = A * B^T where:
//   A: [M, K] activation tensor (float32)
//   B: [N, K] weight matrix stored row-major (transposed for output [M, N])
//   C: [M, N] output tensor
//
// Uses 16x16 tiled approach with shared memory for bank-conflict-free access.
// Each workgroup computes a TILE_M x TILE_N output tile.

// Uniforms: matrix dimensions
struct Params {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;       // [M, K]
@group(0) @binding(2) var<storage, read> B: array<f32>;       // [N, K] (row-major, B^T for matmul)
@group(0) @binding(3) var<storage, read_write> C: array<f32>; // [M, N]

const TILE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;  // [TILE, TILE]
var<workgroup> tileB: array<array<f32, 16>, 16>;  // [TILE, TILE]

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;

    var acc: f32 = 0.0;

    let numTiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A: A[row, t*TILE + lid.x]
        let a_col = t * TILE + lid.x;
        if (row < params.M && a_col < params.K) {
            tileA[lid.y][lid.x] = A[row * params.K + a_col];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }

        // Load tile of B^T: B[col, t*TILE + lid.y] (B stored as [N, K])
        let b_k = t * TILE + lid.y;
        if (col < params.N && b_k < params.K) {
            tileB[lid.y][lid.x] = B[col * params.K + b_k];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        // Accumulate dot product for this tile
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tileA[lid.y][k] * tileB[k][lid.x];
        }

        workgroupBarrier();
    }

    // Store result
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = acc;
    }
}
