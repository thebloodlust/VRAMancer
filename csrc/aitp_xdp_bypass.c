/*
 * VRAMancer AITP XDP Bypass — Kernel-level NIC interception
 *
 * Two modes:
 *   MODE 0 (XSKMAP): Redirect AITP packets to AF_XDP userspace socket for
 *                     zero-copy processing. Userspace then writes to GPU via
 *                     cuMemcpy or nvidia_peermem.  This is the DEFAULT.
 *   MODE 1 (DROP):    Pure kernel processing — count bytes and drop.
 *                     Used when nvidia_peermem is loaded and DMA is handled
 *                     by the driver directly (future GPUDirect path).
 *
 * Compile:
 *   clang -O2 -g -target bpf -c aitp_xdp_bypass.c -o aitp_xdp_bypass.o
 *
 * Load (AF_XDP redirect mode):
 *   ip link set dev <iface> xdpgeneric obj aitp_xdp_bypass.o sec xdp_aitp_bypass
 *
 * The AF_XDP userspace side is implemented in core/network/aitp_receiver.py
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ipv6.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>

#define AITP_PORT 9109
#define AITP_MAGIC_1 'V'
#define AITP_MAGIC_2 'T'

/* --- BPF Maps --- */

/* Stats: bytes intercepted at NIC level */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __u64);
    __uint(max_entries, 4);  /* 0=bytes, 1=packets, 2=drops, 3=redirects */
} aitp_dma_stats SEC(".maps");

/* AF_XDP socket map — userspace registers its XSK FD here (queue index -> XSK fd) */
struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, 64);  /* Up to 64 NIC queues */
} xsks_map SEC(".maps");

/* Config map — runtime mode selection (0=AF_XDP redirect, 1=kernel drop) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, 1);
} aitp_config SEC(".maps");

static __always_inline int parse_aitp(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    /* 1. Ethernet */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return -1;
    if (eth->h_proto != bpf_htons(ETH_P_IPV6))
        return -1;

    /* 2. IPv6 */
    struct ipv6hdr *ipv6 = (void *)(eth + 1);
    if ((void *)(ipv6 + 1) > data_end)
        return -1;
    if (ipv6->nexthdr != IPPROTO_UDP)
        return -1;

    /* 3. UDP */
    struct udphdr *udp = (void *)(ipv6 + 1);
    if ((void *)(udp + 1) > data_end)
        return -1;
    if (udp->dest != bpf_htons(AITP_PORT))
        return -1;

    /* 4. AITP magic "VT" */
    unsigned char *payload = (unsigned char *)(udp + 1);
    if ((void *)(payload + 2) > data_end)
        return -1;
    if (payload[0] != AITP_MAGIC_1 || payload[1] != AITP_MAGIC_2)
        return -1;

    return 0; /* Valid AITP packet */
}

SEC("xdp_aitp_bypass")
int aitp_direct_to_gpu(struct xdp_md *ctx)
{
    if (parse_aitp(ctx) != 0)
        return XDP_PASS;

    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    __u64 pkt_len = data_end - data;

    /* Update stats */
    __u32 key_bytes = 0, key_pkts = 1;
    __u64 *stat;

    stat = bpf_map_lookup_elem(&aitp_dma_stats, &key_bytes);
    if (stat)
        __sync_fetch_and_add(stat, pkt_len);

    stat = bpf_map_lookup_elem(&aitp_dma_stats, &key_pkts);
    if (stat)
        __sync_fetch_and_add(stat, 1);

    /* Check runtime mode */
    __u32 cfg_key = 0;
    __u32 *mode = bpf_map_lookup_elem(&aitp_config, &cfg_key);
    __u32 effective_mode = mode ? *mode : 0;

    if (effective_mode == 0) {
        /* MODE 0: AF_XDP redirect — send to userspace zero-copy socket.
         * Userspace (aitp_receiver.py) reads raw frames from the UMEM ring
         * and writes tensor payload directly to GPU via cuMemcpyHtoD or
         * stages in pinned CPU memory for immediate DMA. */
        __u32 key_redir = 3;
        stat = bpf_map_lookup_elem(&aitp_dma_stats, &key_redir);
        if (stat)
            __sync_fetch_and_add(stat, 1);

        return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, XDP_PASS);
    }

    /* MODE 1: Kernel drop — packet processed entirely in-kernel.
     * Future: with nvidia_peermem loaded, use bpf_xdp_adjust_tail + DMA
     * to write payload directly to GPU BAR1 aperture. */
    __u32 key_drop = 2;
    stat = bpf_map_lookup_elem(&aitp_dma_stats, &key_drop);
    if (stat)
        __sync_fetch_and_add(stat, 1);

    return XDP_DROP;
}

char _license[] SEC("license") = "GPL";
