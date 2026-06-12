"""
AITP Forward Error Correction (FEC) — Real GF(2^8) Reed-Solomon.

Protects UDP tensor transport against packet loss without retransmission.
Uses Cauchy matrices over GF(2^8) for encoding, with Gaussian elimination
for decoding. Can recover from up to `parity_shards` lost data shards.

GF(2^8) arithmetic uses the irreducible polynomial x^8 + x^4 + x^3 + x^2 + 1
(0x11d), which is standard for Reed-Solomon (same as used in RAID-6, QR codes).
"""

import math


# ---------------------------------------------------------------------------
# GF(2^8) arithmetic with lookup tables for speed
# Irreducible polynomial: x^8 + x^4 + x^3 + x^2 + 1 = 0x11d
# ---------------------------------------------------------------------------
_GF_EXP = [0] * 512  # anti-log table (doubled for mod-free wraparound)
_GF_LOG = [0] * 256   # log table

def _init_gf_tables():
    """Build GF(2^8) exp/log lookup tables."""
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d  # reduce by irreducible polynomial
    # Double the exp table for convenient modular access
    for i in range(255, 510):
        _GF_EXP[i] = _GF_EXP[i - 255]

_init_gf_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiply two elements in GF(2^8)."""
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def gf_div(a: int, b: int) -> int:
    """Divide a by b in GF(2^8). b must be nonzero."""
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(2^8)")
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


def gf_inv(a: int) -> int:
    """Multiplicative inverse in GF(2^8)."""
    if a == 0:
        raise ZeroDivisionError("Zero has no inverse in GF(2^8)")
    return _GF_EXP[255 - _GF_LOG[a]]


def _cauchy_matrix(data_shards: int, parity_shards: int) -> list:
    """
    Build a Cauchy encoding matrix over GF(2^8).

    The matrix is (parity_shards x data_shards) where:
        M[i][j] = 1 / (x_i XOR y_j)
    with x_i = i and y_j = parity_shards + j (ensuring x_i != y_j).
    """
    matrix = []
    for i in range(parity_shards):
        row = []
        for j in range(data_shards):
            # Cauchy element: 1/(x_i ^ y_j) in GF(2^8)
            val = i ^ (parity_shards + j)
            if val == 0:
                raise ValueError(
                    f"Cauchy matrix degenerate at ({i},{j}) — "
                    f"reduce shard count (max data+parity = 256)"
                )
            row.append(gf_inv(val))
        matrix.append(row)
    return matrix


def _gf_vec_dot(row: list, columns: list) -> bytes:
    """Dot product of a GF(2^8) coefficient row with byte-array columns."""
    shard_size = len(columns[0])
    result = bytearray(shard_size)
    for j, coeff in enumerate(row):
        if coeff == 0:
            continue
        col = columns[j]
        if coeff == 1:
            for k in range(shard_size):
                result[k] ^= col[k]
        else:
            for k in range(shard_size):
                result[k] ^= gf_mul(coeff, col[k])
    return bytes(result)


class FastFEC:
    """
    Real GF(2^8) Cauchy Reed-Solomon FEC for AITP tensor transport.

    Splits data into `data_shards` fragments, computes `parity_shards`
    distinct parity fragments using a Cauchy matrix. Can reconstruct
    original data from any `data_shards` out of `data_shards + parity_shards`
    received fragments.

    Limits: data_shards + parity_shards <= 256 (GF(2^8) field size).
    """

    def __init__(self, data_shards=10, parity_shards=2):
        if data_shards + parity_shards > 256:
            raise ValueError("Total shards cannot exceed 256 (GF(2^8) limit)")
        if data_shards < 1 or parity_shards < 1:
            raise ValueError("Need at least 1 data shard and 1 parity shard")
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        self.total_shards = data_shards + parity_shards
        self._cauchy = _cauchy_matrix(data_shards, parity_shards)

    def encode(self, tensor_data: bytes) -> list:
        """
        Encode tensor bytes into data_shards + parity_shards fragments.

        Each parity shard is a DISTINCT linear combination of data shards
        computed via the Cauchy matrix over GF(2^8). This allows recovery
        from up to `parity_shards` lost data fragments.

        Returns list of bytes objects: [data_0, ..., data_N, parity_0, ..., parity_M]
        """
        shard_size = math.ceil(len(tensor_data) / self.data_shards)
        padded_len = shard_size * self.data_shards
        padded_data = tensor_data.ljust(padded_len, b'\x00')

        # Split into data shards
        data_chunks = []
        for i in range(self.data_shards):
            start = i * shard_size
            data_chunks.append(padded_data[start:start + shard_size])

        # Compute parity shards via Cauchy matrix
        shards = list(data_chunks)
        for i in range(self.parity_shards):
            parity = _gf_vec_dot(self._cauchy[i], data_chunks)
            shards.append(parity)

        return shards

    def decode(self, received_shards: dict, original_size: int) -> bytes:
        """
        Reconstruct original data from any `data_shards` received fragments.

        Args:
            received_shards: dict mapping shard_index -> shard_bytes.
                Indices 0..data_shards-1 are data shards.
                Indices data_shards..total_shards-1 are parity shards.
            original_size: original byte length before padding.

        Returns the reconstructed original bytes.

        Raises AssertionError if fewer than data_shards fragments received.
        """
        available = sorted(received_shards.keys())
        if len(available) < self.data_shards:
            raise ValueError(
                f"Need at least {self.data_shards} shards to reconstruct, "
                f"got {len(available)}"
            )

        # Use exactly data_shards fragments
        use_indices = available[:self.data_shards]
        shard_size = len(next(iter(received_shards.values())))

        # Check if all data shards are present (fast path — no decoding needed)
        data_indices = set(range(self.data_shards))
        have_data = set(i for i in use_indices if i < self.data_shards)

        if have_data == data_indices:
            # All data shards present — just concatenate
            data = b''
            for i in range(self.data_shards):
                data += received_shards[i]
            return data[:original_size]

        # --- Slow path: build and solve the linear system ---
        # Build the encoding matrix rows for the shards we have.
        # For data shard i: row is the i-th identity row.
        # For parity shard (data_shards + p): row is cauchy[p].
        matrix = []
        rhs_columns = []

        for idx in use_indices:
            if idx < self.data_shards:
                # Identity row
                row = [0] * self.data_shards
                row[idx] = 1
            else:
                # Parity row from Cauchy matrix
                p = idx - self.data_shards
                row = list(self._cauchy[p])
            matrix.append(row)
            rhs_columns.append(bytearray(received_shards[idx]))

        # Gaussian elimination with partial pivoting over GF(2^8)
        n = self.data_shards
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if matrix[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                raise ValueError(f"Singular matrix at column {col} — cannot decode")

            # Swap rows
            if pivot_row != col:
                matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
                rhs_columns[col], rhs_columns[pivot_row] = rhs_columns[pivot_row], rhs_columns[col]

            # Scale pivot row so pivot = 1
            inv_pivot = gf_inv(matrix[col][col])
            for j in range(n):
                matrix[col][j] = gf_mul(matrix[col][j], inv_pivot)
            for k in range(shard_size):
                rhs_columns[col][k] = gf_mul(rhs_columns[col][k], inv_pivot)

            # Eliminate column in all other rows
            for row in range(n):
                if row == col:
                    continue
                factor = matrix[row][col]
                if factor == 0:
                    continue
                for j in range(n):
                    matrix[row][j] ^= gf_mul(factor, matrix[col][j])
                for k in range(shard_size):
                    rhs_columns[row][k] ^= gf_mul(factor, rhs_columns[col][k])

        # rhs_columns now contains the reconstructed data shards in order
        data = b''
        for i in range(n):
            data += bytes(rhs_columns[i])
        return data[:original_size]

