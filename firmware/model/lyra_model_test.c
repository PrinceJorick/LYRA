/* Host test: Python/C parity on golden vectors (ml/lyra/golden.py) for window quality, features,
 * model probability, threshold decision and detector state transitions.
 *
 *   cc -std=c99 -O2 -Wall -Wextra -I firmware/model firmware/model/lyra_model_test.c \
 *      firmware/model/lyra_features.c firmware/model/lyra_detector.c -lm -o lyra_model_test
 *   ./lyra_model_test firmware/model/test/golden_vectors.bin
 *
 * Tolerances: features |c - py| <= 1e-5 * max(1, |py|) (float32 outputs of double maths, different
 * summation order); probability |c - py| <= 1e-6; threshold decisions must match except for rows
 * whose reference probability is within 1e-6 of the threshold; detector flags must match exactly.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lyra_detector.h"
#include "lyra_detector_config.h"
#include "lyra_features.h"
#include "lyra_model.h"

static int rd(void *dst, size_t n, FILE *f) { return fread(dst, 1, n, f) == n; }

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s golden_vectors.bin\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    char magic[8];
    int32_t n_win;
    if (!f || !rd(magic, 8, f) || memcmp(magic, "LYRAGV01", 8) || !rd(&n_win, 4, f)) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    int fails = 0, n_feat = 0, n_prob = 0, exact = 0;
    double max_rel = 0.0, max_prob = 0.0;
    static float env[2 * 1024], acc[3 * 1024], gyr[3 * 1024];
    for (int w = 0; w < n_win; w++) {
        int32_t n, code_v, code_f;
        float ref[LYRA_N_FEATURES], ref_p, x[LYRA_N_FEATURES];
        if (!rd(&n, 4, f) || n <= 0 || n > 1024 || !rd(env, 8 * (size_t)n, f) || !rd(acc, 12 * (size_t)n, f) ||
            !rd(gyr, 12 * (size_t)n, f) || !rd(&code_v, 4, f) || !rd(&code_f, 4, f) || !rd(ref, sizeof ref, f) ||
            !rd(&ref_p, 4, f)) {
            fprintf(stderr, "truncated window %d\n", w);
            return 2;
        }
        const int v = lyra_window_valid(env, 2 * n, acc, gyr, n, INFINITY);
        int c = n <= LYRA_MAX_IMU_SAMPLES ? lyra_features_compute(env, 2 * n, acc, gyr, n, x) : code_f;
        if (n > LYRA_MAX_IMU_SAMPLES) continue; /* longer than the deployed window's scratch buffers */
        if (v != code_v || c != code_f) {
            printf("FAIL window %d: valid %d/%d features %d/%d (c/py)\n", w, v, code_v, c, code_f);
            fails++;
            continue;
        }
        if (c != LYRA_OK) continue;
        for (int i = 0; i < LYRA_N_FEATURES; i++) {
            const double d = fabs((double)x[i] - ref[i]), rel = d / fmax(1.0, fabs(ref[i]));
            n_feat++;
            exact += x[i] == ref[i];
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-5) {
                printf("FAIL window %d feature %s: c %.9g py %.9g\n", w, LYRA_FEATURE_NAMES[i], x[i], ref[i]);
                fails++;
            }
        }
        const float p = lyra_model_predict(ref); /* model parity on identical inputs */
        const double dp = fabs((double)p - ref_p);
        n_prob++;
        if (dp > max_prob) max_prob = dp;
        if (dp > 1e-6 || ((p >= LYRA_THRESHOLD) != (ref_p >= LYRA_THRESHOLD) && fabs(ref_p - LYRA_THRESHOLD) > 1e-6)) {
            printf("FAIL window %d probability: c %.9g py %.9g\n", w, p, ref_p);
            fails++;
        }
    }
    int32_t n_steps, n_open, m_close;
    float thr;
    uint32_t refr, gap;
    if (!rd(&n_steps, 4, f) || !rd(&thr, 4, f) || !rd(&n_open, 4, f) || !rd(&m_close, 4, f) || !rd(&refr, 4, f) ||
        !rd(&gap, 4, f)) {
        fprintf(stderr, "truncated detector header\n");
        return 2;
    }
    const lyra_detector_cfg_t cfg = {thr, (uint8_t)n_open, (uint8_t)m_close, refr, gap};
    lyra_detector_t d;
    lyra_detector_init(&d, &cfg);
    int det_fails = 0;
    for (int s = 0; s < n_steps; s++) {
        uint8_t op;
        float p;
        uint32_t t, want;
        if (!rd(&op, 1, f) || !rd(&p, 4, f) || !rd(&t, 4, f) || !rd(&want, 4, f)) {
            fprintf(stderr, "truncated detector step %d\n", s);
            return 2;
        }
        unsigned got = 0;
        if (op == 1)
            lyra_detector_reset(&d);
        else
            got = lyra_detector_update(&d, p, t);
        if (got != want) {
            printf("FAIL detector step %d: p %.9g t %u flags c %u py %u\n", s, p, t, got, want);
            det_fails++;
        }
    }
    fclose(f);
    printf("windows %d, features compared %d (exact %d, max rel diff %.3g), probabilities %d (max abs diff %.3g), "
           "detector steps %d, failures %d\n", n_win, n_feat, exact, max_rel, n_prob, max_prob, n_steps, fails + det_fails);
    return fails + det_fails ? 1 : 0;
}
