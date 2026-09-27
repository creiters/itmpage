#include <stdint.h>
#include <stdbool.h>

// Definice vestavěné funkce Clangu namísto string.h memcpy
#define memcpy(dest, src, n) __builtin_memcpy(dest, src, n)

#define NULL ((void*)0)
#define RING_BUFFER_CAPACITY 64
#define S7_RX_BUF_SIZE 2048

// Imported Host Syscalls (Provided by WasiWebRTCHost in JavaScript)
__attribute__((import_module("wasi_webrtc"), import_name("send_mesh_broadcast")))
extern void wasi_webrtc_send_broadcast(const uint8_t *data, int32_t len);

__attribute__((import_module("wasi_webrtc"), import_name("notify_peer_timeout")))
extern void wasi_webrtc_notify_peer_timeout(int32_t peer_index);

typedef struct {
    uint64_t timestamp;
    uint32_t station_id;
    uint16_t db_num;
    uint16_t offset;
    float value;
    uint8_t is_replicated; // 0 = local, 1 = remote peer replica
} TelemetryFrame;

static TelemetryFrame g_ring_buffer[RING_BUFFER_CAPACITY];
static uint32_t g_ring_head = 0;
static uint32_t g_ring_count = 0;
static uint8_t g_s7_rx_buffer[S7_RX_BUF_SIZE];

uint8_t* get_s7_rx_buffer(void) {
    return g_s7_rx_buffer;
}

int32_t get_s7_rx_buffer_capacity(void) {
    return S7_RX_BUF_SIZE;
}

float wasi_parse_s7_real(int32_t offset) {
    if (offset < 0 || offset + 4 > S7_RX_BUF_SIZE) return 0.0f;
    uint32_t val = ((uint32_t)g_s7_rx_buffer[offset] << 24) |
                   ((uint32_t)g_s7_rx_buffer[offset + 1] << 16) |
                   ((uint32_t)g_s7_rx_buffer[offset + 2] << 8) |
                   ((uint32_t)g_s7_rx_buffer[offset + 3]);
    float res;
    memcpy(&res, &val, sizeof(res));
    return res;
}

int16_t wasi_parse_s7_int(int32_t offset) {
    if (offset < 0 || offset + 2 > S7_RX_BUF_SIZE) return 0;
    return (int16_t)(((uint16_t)g_s7_rx_buffer[offset] << 8) | g_s7_rx_buffer[offset + 1]);
}

int32_t wasi_parse_s7_dint(int32_t offset) {
    if (offset < 0 || offset + 4 > S7_RX_BUF_SIZE) return 0;
    return (int32_t)(((uint32_t)g_s7_rx_buffer[offset] << 24) |
                     ((uint32_t)g_s7_rx_buffer[offset + 1] << 16) |
                     ((uint32_t)g_s7_rx_buffer[offset + 2] << 8) |
                     g_s7_rx_buffer[offset + 3]);
}

uint8_t wasi_parse_s7_byte(int32_t offset) {
    if (offset < 0 || offset >= S7_RX_BUF_SIZE) return 0;
    return g_s7_rx_buffer[offset];
}

int32_t wasi_push_local_sample(uint64_t timestamp, uint32_t station_id, uint16_t db, uint16_t off, float val) {
    uint32_t idx = (g_ring_head + g_ring_count) % RING_BUFFER_CAPACITY;
    g_ring_buffer[idx].timestamp = timestamp;
    g_ring_buffer[idx].station_id = station_id;
    g_ring_buffer[idx].db_num = db;
    g_ring_buffer[idx].offset = off;
    g_ring_buffer[idx].value = val;
    g_ring_buffer[idx].is_replicated = 0;

    if (g_ring_count < RING_BUFFER_CAPACITY) {
        g_ring_count++;
    } else {
        g_ring_head = (g_ring_head + 1) % RING_BUFFER_CAPACITY;
    }

    // Binary payload: [Timestamp 8B][StationID 4B][DB 2B][Offset 2B][Value 4B][ReplicaFlag 1B] = 21 Bytes
    uint8_t serialized[21];
    memcpy(&serialized[0], &timestamp, 8);
    memcpy(&serialized[8], &station_id, 4);
    memcpy(&serialized[12], &db, 2);
    memcpy(&serialized[14], &off, 2);
    memcpy(&serialized[16], &val, 4);
    serialized[20] = 1;

    wasi_webrtc_send_broadcast(serialized, 21);
    return idx;
}

int32_t wasi_receive_mesh_frame(const uint8_t *data, int32_t len) {
    if (len < 21) return -1;

    uint32_t idx = (g_ring_head + g_ring_count) % RING_BUFFER_CAPACITY;
    memcpy(&g_ring_buffer[idx].timestamp, &data[0], 8);
    memcpy(&g_ring_buffer[idx].station_id, &data[8], 4);
    memcpy(&g_ring_buffer[idx].db_num, &data[12], 2);
    memcpy(&g_ring_buffer[idx].offset, &data[14], 2);
    memcpy(&g_ring_buffer[idx].value, &data[16], 4);
    g_ring_buffer[idx].is_replicated = 1;

    if (g_ring_count < RING_BUFFER_CAPACITY) {
        g_ring_count++;
    } else {
        g_ring_head = (g_ring_head + 1) % RING_BUFFER_CAPACITY;
    }

    return idx;
}

int32_t wasi_get_ring_count(void) {
    return g_ring_count;
}

TelemetryFrame* wasi_get_frame_ptr(int32_t relative_idx) {
    if (relative_idx < 0 || (uint32_t)relative_idx >= g_ring_count) return NULL;
    uint32_t actual_idx = (g_ring_head + relative_idx) % RING_BUFFER_CAPACITY;
    return &g_ring_buffer[actual_idx];
}

void wasi_flush_ring(void) {
    g_ring_head = 0;
    g_ring_count = 0;
}
