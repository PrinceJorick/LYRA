/* lyra_features.h: C99 port of lyra_mlm.features for the 22-feature multimodal schema
 * (50 Hz envelope + 25 Hz 6-axis IMU), in exactly the training feature order.
 *
 * EXPERIMENTAL research code, not a medical device. Parity with the Python reference is checked by
 * mlm/tests/test_c99_port.py.
 *
 * Preconditions (the firmware suppresses any window that breaks them, so it is never scored):
 *   - every sample is usable: no missing or invalid samples;
 *   - timestamps are strictly ascending, regular and gap-free;
 *   - both streams cover the same window (e.g. 4 s);
 *   - the window's quality gate passed. This function does not assess quality.
 * The envelope and IMU timestamps are passed in, because the envelope-to-IMU alignment compares
 * them exactly as the Python reference does.
 */
#ifndef LYRA_FEATURES_H
#define LYRA_FEATURES_H

#define LYRA_N_FEATURES 22
#define LYRA_N_EMG_FEATURES 9  /* [0, 9): SIMULATED_ENV / ENV block */
#define LYRA_N_IMU_FEATURES 9  /* [9, 18): IMU block; [18, 22): cross-modal */
#define LYRA_MAX_SAMPLES 256   /* per stream per window: 4 s at 50 Hz = 200 */

/* Feature order (index: name):
 *  0 emg_simenv_rms            1 emg_simenv_mav             2 emg_simenv_var
 *  3 emg_simenv_std            4 emg_simenv_wl_per_s        5 emg_simenv_autocorr_peak
 *  6 emg_simenv_mod_band_power 7 emg_simenv_mod_band_ratio  8 emg_simenv_mod_peak_freq_hz
 *  9 imu_acc_mag_mean         10 imu_acc_mag_std           11 imu_gyr_mag_mean
 * 12 imu_gyr_mag_std          13 imu_gyr_mag_rms           14 imu_acc_jerk_rms
 * 15 imu_acc_dominant_freq_hz 16 imu_acc_band_power        17 imu_orientation_change_deg
 * 18 xmod_env_accmag_corr     19 xmod_energy_log_ratio     20 xmod_lag_s
 * 21 xmod_coherence_band
 * Units: ACC in g, GYR in deg/s, envelope in sensor units.
 */

/* env_t/env: n_env envelope samples at env_rate_hz (n_env may be 0: EMG and cross-modal blocks NaN).
 * imu_t/imu: n_imu IMU samples at imu_rate_hz, imu row-major [ax ay az gx gy gz] per sample
 * (n_imu may be 0: IMU and cross-modal blocks NaN).
 * out: LYRA_N_FEATURES values; NaN marks an unavailable feature (never 0).
 * Returns 0, or -1 for invalid arguments. Not reentrant: uses static scratch buffers. */
int lyra_features(const double *env_t, const double *env, int n_env, double env_rate_hz,
                  const double *imu_t, const double *imu, int n_imu, double imu_rate_hz,
                  double out[LYRA_N_FEATURES]);

#endif
