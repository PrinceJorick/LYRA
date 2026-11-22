/* lyra_logline.c: see lyra_logline.h. */
#include "lyra_logline.h"

#include <stdio.h>

static int fits(int r, size_t n) { return r > 0 && (size_t)r < n ? r : -1; }

static int token_ok(const char *s) {  /* ids and labels: [A-Za-z0-9._-]+, never a separator */
    if (s == NULL || *s == '\0') return 0;
    for (; *s; ++s) {
        char c = *s;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
              c == '.'))
            return 0;
    }
    return 1;
}

int lyra_log_header(char *buf, size_t n, const char *pod_id, const char *firmware, double env_rate_hz,
                    double raw_rate_hz, double imu_rate_hz, double full_scale_mv, int acc_range_g, int gyr_range_dps) {
    if (buf == NULL || !token_ok(pod_id) || !token_ok(firmware)) return -1;
    return fits(snprintf(buf, n, "S,schema=1,pod_id=%s,firmware=%s,env_rate_hz=%g,raw_rate_hz=%g,imu_rate_hz=%g,"
                                 "full_scale_mv=%g,acc_range_g=%d,gyr_range_dps=%d\n",
                         pod_id, firmware, env_rate_hz, raw_rate_hz, imu_rate_hz, full_scale_mv, acc_range_g,
                         gyr_range_dps), n);
}

int lyra_log_emg(char *buf, size_t n, char kind, double t_s, double mv, uint8_t flag) {
    if (buf == NULL || (kind != 'E' && kind != 'R') || flag > 2) return -1;
    return fits(snprintf(buf, n, "%c,%.6f,%.3f,%u\n", kind, t_s, mv, (unsigned)flag), n);
}

int lyra_log_imu(char *buf, size_t n, double t_s, const double v[6], uint8_t flag) {
    if (buf == NULL || v == NULL || flag > 2) return -1;
    return fits(snprintf(buf, n, "I,%.6f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%u\n", t_s, v[0], v[1], v[2], v[3], v[4], v[5],
                         (unsigned)flag), n);
}

int lyra_log_marker(char *buf, size_t n, double t_s, const char *label) {
    if (buf == NULL || !token_ok(label)) return -1;
    return fits(snprintf(buf, n, "M,%.6f,%s\n", t_s, label), n);
}
