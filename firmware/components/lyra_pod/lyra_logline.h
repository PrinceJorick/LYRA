/* lyra_logline.h: data-collection mode line format (pod -> USB serial -> laptop logger).
 *
 * One ASCII line per record. mlm/scripts/pod_logger.py turns a capture into a session folder in the
 * mlm/docs/POD_DATA_PROTOCOL.md format (schema v1):
 *   S,schema=1,pod_id=<id>,firmware=<ver>,env_rate_hz=50,raw_rate_hz=1000,imu_rate_hz=25,
 *     full_scale_mv=3300,acc_range_g=8,gyr_range_dps=1000     session header (first line)
 *   E,<t_s>,<mV>,<flag>                    MyoWare ENV
 *   R,<t_s>,<mV>,<flag>                    MyoWare RAW
 *   I,<t_s>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<flag>   IMU (g, deg/s)
 *   M,<t_s>,<label>                        marker (e.g. pod button), written to markers.tsv
 * Times are seconds on the pod clock from session start. Participant, consent and placement are
 * entered on the laptop, never on the pod. A malformed line is dropped by the logger and counted.
 * Each function returns the line length (snprintf), or -1 if it does not fit / bad arguments.
 */
#ifndef LYRA_LOGLINE_H
#define LYRA_LOGLINE_H

#include <stddef.h>
#include <stdint.h>

#define LYRA_LOGLINE_MAX 160

int lyra_log_header(char *buf, size_t n, const char *pod_id, const char *firmware, double env_rate_hz,
                    double raw_rate_hz, double imu_rate_hz, double full_scale_mv, int acc_range_g, int gyr_range_dps);
int lyra_log_emg(char *buf, size_t n, char kind /* 'E' or 'R' */, double t_s, double mv, uint8_t flag);
int lyra_log_imu(char *buf, size_t n, double t_s, const double v[6], uint8_t flag);
int lyra_log_marker(char *buf, size_t n, double t_s, const char *label /* [a-z0-9_-]+ */);

#endif
