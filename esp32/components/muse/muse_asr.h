/*
 * Speech-to-text on your own server (tools/muse_asr), for boards that can't
 * upload a voice note to Muse: push-to-talk streams the mic there, and the
 * transcript goes to Muse as a text message. One turn at a time, voice task only.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Connects and starts the upload. False if the server can't be reached. */
bool muse_asr_begin(void);
/* Queues mic audio (16 kHz mono); sends it as whole frames fill. */
bool muse_asr_audio(const int16_t *pcm, size_t frames);
/* Sends what's left and the end of speech; the transcript follows. */
bool muse_asr_end(void);
/* 1: the transcript is in `text` (possibly empty), 0: not yet, -1: failed. */
int muse_asr_poll(char *text, size_t cap);
/* Ends the turn's connection and frees its buffers; safe to repeat. */
void muse_asr_close(void);
