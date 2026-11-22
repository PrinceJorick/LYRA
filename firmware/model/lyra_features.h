/* LYRA window features for the ESP32-S3 pod (C99, no dependencies beyond libm).
 *
 * Experimental research component: not clinically validated, not a diagnostic device.
 *
 * Inputs (locked decision D12): one completed window of
 *   env : MyoWare 2.0 ENV at 50 Hz, n_env = 2 * n_imu samples, uV-equivalent units
 *         (the ADC conversion must apply the gain calibration described in models/lyra/README.md)
 *   acc : BMI270 accelerometer at 25 Hz, n_imu x 3 interleaved (x,y,z), m/s^2
 *   gyr : BMI270 gyroscope at 25 Hz, n_imu x 3 interleaved, rad/s
 * Feature order, units and definitions: lyra_feature_spec.h (generated from ml/lyra/feature_spec.json).
 * The Python reference is ml/lyra/features.py; parity is tested by lyra_model_test.c.
 *
 * Not reentrant: uses static scratch buffers (counted in .bss, see reports/lyra/resource_report.json).
 */
#ifndef LYRA_FEATURES_H
#define LYRA_FEATURES_H

#include "lyra_feature_spec.h"

#define LYRA_OK 0
#define LYRA_ERR_SIZE (-1)      /* n_env != 2*n_imu, window too short/long */
#define LYRA_ERR_NONFINITE (-2) /* NaN/Inf input sample */
#define LYRA_ERR_FLAT (-3)      /* ENV or all IMU channels constant (sensor off/disconnected) */
#define LYRA_ERR_SATURATED (-4) /* ENV at/above the configured ADC saturation level in > 5% of samples */

/* Device-side window quality. env_saturation_level: ENV value treated as ADC full scale
 * (use INFINITY until the pod's ADC range is calibrated). Returns LYRA_OK or an error code. */
int lyra_window_valid(const float *env, int n_env, const float *acc, const float *gyr, int n_imu,
                      float env_saturation_level);

/* Computes the LYRA_N_FEATURES features in spec order into out. Returns LYRA_OK or an error code
 * (then out is untouched and the window must be treated as invalid: no score, detector holds). */
int lyra_features_compute(const float *env, int n_env, const float *acc, const float *gyr, int n_imu,
                          float out[LYRA_N_FEATURES]);

#endif
