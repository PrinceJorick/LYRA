/* lyra_ble.h: pod-side BLE payload codec, byte-exact with mlm/src/lyra_mlm/ble.py (draft protocol v1,
 * mlm/docs/BLE_PROTOCOL.md). Little-endian, explicit byte packing (no struct layout assumptions).
 * Every payload starts with version (1) and origin (1 = HARDWARE, 2 = SIMULATED). The phone must
 * show simulated payloads as simulated, never as hardware readings. */
#ifndef LYRA_BLE_H
#define LYRA_BLE_H

#include <stdint.h>

#define LYRA_BLE_VERSION 1
#define LYRA_BLE_DEVICE_STATUS_LEN 13
#define LYRA_BLE_EVENT_STATUS_LEN 11
#define LYRA_BLE_DEVICE_CONFIG_LEN 10
#define LYRA_BLE_PHONE_COMMAND_LEN 5

/* GATT layout: DRAFT, pending owner approval. Same strings as lyra_mlm.ble.GATT_UUIDS. */
#define LYRA_GATT_SERVICE_UUID "28b70001-8eb8-490a-aaa2-b981ca09340c"
#define LYRA_GATT_DEVICE_STATUS_UUID "28b70002-8eb8-490a-aaa2-b981ca09340c" /* read + notify */
#define LYRA_GATT_EVENT_STATUS_UUID "28b70003-8eb8-490a-aaa2-b981ca09340c"  /* read + indicate */
#define LYRA_GATT_DEVICE_CONFIG_UUID "28b70004-8eb8-490a-aaa2-b981ca09340c" /* read + write (response) */
#define LYRA_GATT_PHONE_COMMAND_UUID "28b70005-8eb8-490a-aaa2-b981ca09340c" /* write (response) */

enum { LYRA_ORIGIN_HARDWARE = 1, LYRA_ORIGIN_SIMULATED = 2 };
enum { LYRA_CANCEL_NONE = 0, LYRA_CANCEL_ON_POD = 1, LYRA_CANCEL_ON_PHONE = 2 };
enum { LYRA_COMMAND_CANCEL_EVENT = 1 }; /* phone -> pod (DRAFT) */

typedef struct {
    uint8_t origin;
    uint8_t command;
    uint16_t event_id;
} lyra_ble_command;

typedef struct {
    uint8_t origin;
    uint16_t emg_rate_hz;
    uint8_t imu_odr_hz;  /* 25 (initial) or 50 (experimental) */
    uint16_t window_ms;
    uint16_t step_ms;
    uint8_t countdown_s; /* 7..10 */
} lyra_ble_config;

/* Battery (MAX17048), per-modality quality codes (0..4), heartbeat and uptime. Returns 0, or -1 when
 * a field is out of range (battery 0..100 %, origin 1..2, quality 0..4). */
int lyra_ble_pack_device_status(uint8_t out[LYRA_BLE_DEVICE_STATUS_LEN], uint8_t origin, uint8_t battery_percent,
                                uint16_t battery_mv, uint8_t emg_quality, uint8_t imu_quality, uint16_t heartbeat_seq,
                                uint32_t uptime_s);

/* Possible-event state (0 IDLE, 1 ACCUMULATING, 2 POSSIBLE_EVENT, 3 REFRACTORY), cancellation (0..2),
 * pod uptime at the change, usable modalities (bit 0 EMG, bit 1 IMU). Returns 0 or -1. */
int lyra_ble_pack_event_status(uint8_t out[LYRA_BLE_EVENT_STATUS_LEN], uint8_t origin, uint16_t event_id,
                               uint8_t state, uint8_t cancellation, uint32_t timestamp_s, uint8_t available_modalities);

/* Phone -> pod configuration. Returns 0, or -1 for a wrong length, version, origin or range. */
int lyra_ble_unpack_device_config(const uint8_t *in, int len, lyra_ble_config *out);

/* Phone -> pod command "<BBBH" (DRAFT). Returns 0, or -1 for a wrong length, version, origin or command. */
int lyra_ble_unpack_phone_command(const uint8_t *in, int len, lyra_ble_command *out);

#endif
