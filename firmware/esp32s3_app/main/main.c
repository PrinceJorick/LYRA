/* Size probe: links the LYRA package into an ESP32-S3 app so the linker map shows its real cost.
 * Synthetic input only; not a product firmware. */
#include <stdio.h>

#include "lyra_detector.h"
#include "lyra_detector_config.h"
#include "lyra_features.h"
#include "lyra_model.h"

static float s_env[LYRA_WINDOW_ENV_SAMPLES];
static float s_acc[3 * LYRA_WINDOW_IMU_SAMPLES];
static float s_gyr[3 * LYRA_WINDOW_IMU_SAMPLES];

void app_main(void) {
    float x[LYRA_N_FEATURES];
    for (int i = 0; i < LYRA_WINDOW_ENV_SAMPLES; i++) s_env[i] = 10.0f + (float)(i % 7);
    for (int j = 0; j < LYRA_WINDOW_IMU_SAMPLES; j++) {
        s_acc[3 * j + 2] = 9.81f + 0.01f * (float)(j % 3);
        s_gyr[3 * j] = 0.01f * (float)(j % 5);
    }
    lyra_init_event_state();
    if (lyra_window_valid(s_env, LYRA_WINDOW_ENV_SAMPLES, s_acc, s_gyr, LYRA_WINDOW_IMU_SAMPLES, 4095.0f) == LYRA_OK &&
        lyra_features_compute(s_env, LYRA_WINDOW_ENV_SAMPLES, s_acc, s_gyr, LYRA_WINDOW_IMU_SAMPLES, x) == LYRA_OK) {
        const float p = lyra_model_predict(x);
        printf("p=%f flags=%u\n", (double)p, lyra_update_event_state(p, LYRA_WINDOW_MS));
    }
}
