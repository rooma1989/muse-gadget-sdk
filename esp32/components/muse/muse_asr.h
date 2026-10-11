/*
 * Speech on your own server (tools/muse_asr), for boards that can't upload a
 * voice note to Muse or play its replies: push-to-talk streams the mic there
 * and the transcript goes to Muse as a text message; the reply's text comes
 * back as speech. One of each at a time, voice task only.
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

/* Asks for `text` (UTF-8, Markdown is fine) spoken in `voice` (the server's
 * list). False if the server can't be reached. */
bool muse_tts_begin(const char *text, size_t len, int voice);
/* Up to `frames` of 16 kHz mono speech that has arrived; 0 if none yet. */
size_t muse_tts_read(int16_t *pcm, size_t frames);
/* 0: more speech is coming, 1: all of it has been read (or none started), -1: failed. */
int muse_tts_state(void);
/* About where in the text the speech read so far has got, in bytes. */
size_t muse_tts_text_at(void);
/* Drops the speech and frees its buffers; safe to repeat. */
void muse_tts_close(void);
