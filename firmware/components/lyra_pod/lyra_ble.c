/* lyra_ble.c: see lyra_ble.h. Field order and sizes follow ble.py FORMAT strings:
 * DeviceStatus "<BBBHBBHI", EventStatus "<BBHBBIB", DeviceConfig "<BBHBHHB". */
#include "lyra_ble.h"

#include <stddef.h>

static uint8_t *put16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)(v >> 8);
    return p + 2;
}

static uint8_t *put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)(v >> 24);
    return p + 4;
}

static uint16_t get16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static int origin_ok(uint8_t o) { return o == LYRA_ORIGIN_HARDWARE || o == LYRA_ORIGIN_SIMULATED; }

int lyra_ble_pack_device_status(uint8_t out[LYRA_BLE_DEVICE_STATUS_LEN], uint8_t origin, uint8_t battery_percent,
                                uint16_t battery_mv, uint8_t emg_quality, uint8_t imu_quality, uint16_t heartbeat_seq,
                                uint32_t uptime_s) {
    uint8_t *p = out;
    if (out == NULL || !origin_ok(origin) || battery_percent > 100 || emg_quality > 4 || imu_quality > 4) return -1;
    *p++ = LYRA_BLE_VERSION;
    *p++ = origin;
    *p++ = battery_percent;
    p = put16(p, battery_mv);
    *p++ = emg_quality;
    *p++ = imu_quality;
    p = put16(p, heartbeat_seq);
    put32(p, uptime_s);
    return 0;
}

int lyra_ble_pack_event_status(uint8_t out[LYRA_BLE_EVENT_STATUS_LEN], uint8_t origin, uint16_t event_id,
                               uint8_t state, uint8_t cancellation, uint32_t timestamp_s, uint8_t available_modalities) {
    uint8_t *p = out;
    if (out == NULL || !origin_ok(origin) || state > 3 || cancellation > 2 || available_modalities > 3) return -1;
    *p++ = LYRA_BLE_VERSION;
    *p++ = origin;
    p = put16(p, event_id);
    *p++ = state;
    *p++ = cancellation;
    p = put32(p, timestamp_s);
    *p = available_modalities;
    return 0;
}

int lyra_ble_unpack_phone_command(const uint8_t *in, int len, lyra_ble_command *out) {
    if (in == NULL || out == NULL || len != LYRA_BLE_PHONE_COMMAND_LEN || in[0] != LYRA_BLE_VERSION || !origin_ok(in[1]) ||
        in[2] != LYRA_COMMAND_CANCEL_EVENT)
        return -1;
    out->origin = in[1];
    out->command = in[2];
    out->event_id = get16(in + 3);
    return 0;
}

int lyra_ble_unpack_device_config(const uint8_t *in, int len, lyra_ble_config *out) {
    lyra_ble_config c;
    if (in == NULL || out == NULL || len != LYRA_BLE_DEVICE_CONFIG_LEN || in[0] != LYRA_BLE_VERSION || !origin_ok(in[1]))
        return -1;
    c.origin = in[1];
    c.emg_rate_hz = get16(in + 2);
    c.imu_odr_hz = in[4];
    c.window_ms = get16(in + 5);
    c.step_ms = get16(in + 7);
    c.countdown_s = in[9];
    if (c.emg_rate_hz < 1 || c.window_ms < 1 || c.step_ms < 1 || c.step_ms > c.window_ms ||
        (c.imu_odr_hz != 25 && c.imu_odr_hz != 50) || c.countdown_s < 7 || c.countdown_s > 10)
        return -1;
    *out = c;
    return 0;
}
