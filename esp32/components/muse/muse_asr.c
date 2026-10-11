/*
 * Push-to-talk, and reading replies aloud, through your own speech server.
 * See muse_asr.h, and tools/muse_asr/protocol.py for the wire format: a
 * chunked HTTP POST of a 16-byte hello and AES-256-GCM frames of PCM, answered
 * by one sealed frame holding {"text": ...}; and for speech, a POST of one
 * sealed text frame answered by a stream of sealed IMA ADPCM. Plain TCP to a
 * nearby server: no second TLS session, which a board without PSRAM has no
 * room for beside Link's.
 */
#include "sdkconfig.h"

#if CONFIG_MUSE_ASR_ENABLE

#include "muse_asr.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "cJSON.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "muse_adpcm.h"
#include "lwip/sockets.h"
#include "psa/crypto.h"

static const char *TAG = "muse_asr";

#define RATE 16000
#define FRAME_PCM 1600              /* 50 ms of PCM per frame */
#define TAG_LEN 16
#define HELLO_LEN 16
#define T_AUDIO 1
#define T_END 2
#define T_RESULT 3
#define T_TEXT 4
#define T_SEGMENT 5
#define SERVER_COUNTER 0x80000000u
#define CONNECT_MS 1500
#define SEND_MS 1000
#define RX_CAP 1536
/* chunk header "xxxx\r\n" + u16 length + type + PCM + tag + "\r\n" */
#define OUT_CAP (8 + 2 + 1 + FRAME_PCM + TAG_LEN + 2)

static psa_key_id_t s_key;
static int s_sock = -1;
static uint8_t s_hello[HELLO_LEN];
static uint32_t s_counter;
static uint8_t *s_plain;            /* type byte + PCM being gathered */
static size_t s_fill;               /* PCM bytes in s_plain */
static uint8_t *s_out;
static char *s_rx;
static size_t s_rx_len;

static bool load_key(void)
{
    if (s_key) return true;
    const char *hex = CONFIG_MUSE_ASR_KEY;
    uint8_t key[32];
    if (strlen(hex) != 64) {
        ESP_LOGE(TAG, "CONFIG_MUSE_ASR_KEY must be 64 hex digits");
        return false;
    }
    for (int i = 0; i < 32; i++) {
        unsigned v;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1) return false;
        key[i] = (uint8_t)v;
    }
    if (psa_crypto_init() != PSA_SUCCESS) return false;
    psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
    psa_set_key_usage_flags(&attr, PSA_KEY_USAGE_ENCRYPT | PSA_KEY_USAGE_DECRYPT);
    psa_set_key_algorithm(&attr, PSA_ALG_GCM);
    psa_set_key_type(&attr, PSA_KEY_TYPE_AES);
    psa_set_key_bits(&attr, 256);
    psa_status_t st = psa_import_key(&attr, key, sizeof(key), &s_key);
    memset(key, 0, sizeof(key));
    if (st != PSA_SUCCESS) {
        ESP_LOGE(TAG, "key import failed: %d", (int)st);
        s_key = 0;
        return false;
    }
    return true;
}

static void nonce(const uint8_t *hello, uint32_t counter, uint8_t out[12])
{
    memcpy(out, hello + 4, 8);
    out[8] = counter >> 24;
    out[9] = counter >> 16;
    out[10] = counter >> 8;
    out[11] = counter;
}

static bool send_on(int sock, const void *data, size_t len)
{
    const uint8_t *p = data;
    while (len) {
        int n = send(sock, p, len, 0);
        if (n <= 0) {
            ESP_LOGW(TAG, "send failed: errno %d", errno);
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

static bool send_all(const void *data, size_t len)
{
    return send_on(s_sock, data, len);
}

/* Sends `len` bytes as one HTTP chunk; the bytes sit at s_out + 8. */
static bool send_chunk(size_t len)
{
    char head[9];
    int h = snprintf(head, sizeof(head), "%x\r\n", (unsigned)len);
    uint8_t *start = s_out + 8 - h;
    memcpy(start, head, h);
    memcpy(s_out + 8 + len, "\r\n", 2);
    return send_all(start, h + len + 2);
}

/* Seals s_plain[0..1+s_fill) as the next frame and sends it. */
static bool send_frame(uint8_t type)
{
    s_plain[0] = type;
    uint8_t n[12];
    nonce(s_hello, s_counter++, n);
    size_t ct_len = 0;
    uint8_t *ct = s_out + 8 + 2;
    psa_status_t st = psa_aead_encrypt(s_key, PSA_ALG_GCM, n, sizeof(n), s_hello, HELLO_LEN,
                                       s_plain, 1 + s_fill, ct, 1 + FRAME_PCM + TAG_LEN, &ct_len);
    s_fill = 0;
    if (st != PSA_SUCCESS) {
        ESP_LOGE(TAG, "seal failed: %d", (int)st);
        return false;
    }
    s_out[8] = ct_len & 0xff;
    s_out[9] = ct_len >> 8;
    return send_chunk(2 + ct_len);
}

/* A blocking TCP socket to the server with a send timeout, or -1. */
static int open_socket(void)
{
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(CONFIG_MUSE_ASR_PORT) };
    if (inet_aton(CONFIG_MUSE_ASR_IP, &addr.sin_addr) == 0) {
        ESP_LOGE(TAG, "bad CONFIG_MUSE_ASR_IP");
        return -1;
    }
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) return -1;
    int fl = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, fl | O_NONBLOCK);
    int rc = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
    if (rc < 0 && errno == EINPROGRESS) {
        fd_set w;
        FD_ZERO(&w);
        FD_SET(sock, &w);
        struct timeval tv = { CONNECT_MS / 1000, (CONNECT_MS % 1000) * 1000 };
        int err = 0;
        socklen_t el = sizeof(err);
        rc = select(sock + 1, NULL, &w, NULL, &tv) == 1
             && getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &el) == 0 && err == 0 ? 0 : -1;
    }
    if (rc < 0) {
        ESP_LOGW(TAG, "can't connect to %s:%d", CONFIG_MUSE_ASR_IP, CONFIG_MUSE_ASR_PORT);
        close(sock);
        return -1;
    }
    fcntl(sock, F_SETFL, fl & ~O_NONBLOCK);
    struct timeval tv = { SEND_MS / 1000, (SEND_MS % 1000) * 1000 };
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int one = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return sock;
}

static const char *host_header(void)
{
    return CONFIG_MUSE_ASR_HOST[0] ? CONFIG_MUSE_ASR_HOST : CONFIG_MUSE_ASR_IP;
}

static void new_hello(uint8_t hello[HELLO_LEN])
{
    memcpy(hello, "MSR1", 4);
    esp_fill_random(hello + 4, 8);
    hello[12] = RATE & 0xff;
    hello[13] = (RATE >> 8) & 0xff;
    hello[14] = 0;
    hello[15] = 0;
}

bool muse_asr_begin(void)
{
    muse_asr_close();
    if (!load_key()) return false;
    s_plain = malloc(1 + FRAME_PCM);
    s_out = malloc(OUT_CAP);
    if (!s_plain || !s_out || (s_sock = open_socket()) < 0) {
        muse_asr_close();
        return false;
    }
    char req[200];
    int n = snprintf(req, sizeof(req),
                     "POST %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/octet-stream\r\n"
                     "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                     CONFIG_MUSE_ASR_PATH, host_header());
    new_hello(s_hello);
    s_counter = 0;
    s_fill = 0;
    memcpy(s_out + 8, s_hello, HELLO_LEN);
    if (!send_all(req, n) || !send_chunk(HELLO_LEN)) {
        muse_asr_close();
        return false;
    }
    ESP_LOGI(TAG, "streaming to %s:%d", CONFIG_MUSE_ASR_IP, CONFIG_MUSE_ASR_PORT);
    return true;
}

bool muse_asr_audio(const int16_t *pcm, size_t frames)
{
    if (s_sock < 0) return false;
    const uint8_t *p = (const uint8_t *)pcm;
    size_t n = frames * 2;
    while (n) {
        size_t take = FRAME_PCM - s_fill < n ? FRAME_PCM - s_fill : n;
        memcpy(s_plain + 1 + s_fill, p, take);
        s_fill += take;
        p += take;
        n -= take;
        if (s_fill == FRAME_PCM && !send_frame(T_AUDIO)) return false;
    }
    return true;
}

bool muse_asr_end(void)
{
    if (s_sock < 0) return false;
    if (s_fill && !send_frame(T_AUDIO)) return false;
    if (!send_frame(T_END) || !send_all("0\r\n\r\n", 5)) return false;
    free(s_plain);
    s_plain = NULL;
    s_rx = malloc(RX_CAP + 1);
    s_rx_len = 0;
    int fl = fcntl(s_sock, F_GETFL, 0);
    fcntl(s_sock, F_SETFL, fl | O_NONBLOCK);
    return s_rx != NULL;
}

/* Opens the server's sealed result frame into `text`. */
static int open_result(const uint8_t *body, size_t len, char *text, size_t cap)
{
    if (len < 2) return -1;
    size_t ct_len = body[0] | body[1] << 8;
    if (ct_len + 2 > len || ct_len <= TAG_LEN) return -1;
    uint8_t n[12];
    nonce(s_hello, SERVER_COUNTER, n);
    size_t plain_len = 0;
    /* Decrypt in place over the tail of s_out: no new buffer. */
    uint8_t *plain = s_out;
    if (ct_len - TAG_LEN + 1 > OUT_CAP) return -1;
    if (psa_aead_decrypt(s_key, PSA_ALG_GCM, n, sizeof(n), s_hello, HELLO_LEN, body + 2, ct_len,
                         plain, OUT_CAP - 1, &plain_len) != PSA_SUCCESS || plain_len < 1
        || plain[0] != T_RESULT) {
        ESP_LOGW(TAG, "result didn't open");
        return -1;
    }
    plain[plain_len] = '\0';
    cJSON *root = cJSON_Parse((const char *)plain + 1);
    const char *t = cJSON_GetStringValue(cJSON_GetObjectItem(root, "text"));
    strlcpy(text, t ? t : "", cap);
    cJSON_Delete(root);
    return t ? 1 : -1;
}

int muse_asr_poll(char *text, size_t cap)
{
    if (s_sock < 0 || !s_rx) return -1;
    for (;;) {
        if (s_rx_len >= RX_CAP) return -1;
        int n = recv(s_sock, s_rx + s_rx_len, RX_CAP - s_rx_len, 0);
        if (n > 0) {
            s_rx_len += n;
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        break;  /* closed: whatever arrived is the whole reply */
    }
    s_rx[s_rx_len] = '\0';
    char *body = strstr(s_rx, "\r\n\r\n");
    if (!body) return 0;
    body += 4;
    int status = 0;
    sscanf(s_rx, "HTTP/1.%*d %d", &status);
    const char *cl = NULL;
    for (const char *h = s_rx; h < body && !cl; h++) {
        if (*h == '\n' && !strncasecmp(h + 1, "content-length:", 15)) cl = h;
    }
    size_t want = cl ? strtoul(cl + 16, NULL, 10) : 0;
    size_t have = s_rx_len - (body - s_rx);
    if (status != 200) {
        ESP_LOGW(TAG, "server answered %d", status);
        return -1;
    }
    if (!cl || have < want) return 0;
    return open_result((const uint8_t *)body, want, text, cap);
}

void muse_asr_close(void)
{
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
    }
    free(s_plain);
    free(s_out);
    free(s_rx);
    s_plain = NULL;
    s_out = NULL;
    s_rx = NULL;
    s_fill = 0;
    s_rx_len = 0;
}

/* ---- Reading a reply aloud ---- */

#define TTS_ADPCM_MAX 2048          /* ADPCM bytes per frame, as the server sends */
#define TTS_PLAIN_CAP (1 + TTS_ADPCM_MAX)
#define TTS_RX_CAP (2 + TTS_PLAIN_CAP + TAG_LEN + 384)   /* a frame, or the response head */
#define TTS_TEXT_MAX 1200

static struct {
    int sock;
    uint8_t hello[HELLO_LEN];
    uint32_t counter;
    uint8_t *rx;
    size_t rx_len;
    bool head;                      /* the HTTP head is behind us */
    uint8_t *plain;                 /* the frame being decoded: type byte + ADPCM */
    size_t plain_len, plain_off;
    muse_adpcm_t adpcm;
    bool done, failed;
    /* The sentence being said: its samples and where it sits in the text. */
    uint32_t seg_base, seg_samples, delivered;
    uint16_t seg_from, seg_to;
    int64_t t0;
} s_tts = { .sock = -1 };

static void tts_fail(const char *why)
{
    if (!s_tts.failed) ESP_LOGW(TAG, "speech: %s", why);
    s_tts.failed = true;
}

bool muse_tts_begin(const char *text, size_t len, int voice)
{
    muse_tts_close();
    if (!CONFIG_MUSE_TTS_PATH[0] || !load_key()) return false;
    if (len > TTS_TEXT_MAX) {
        len = TTS_TEXT_MAX;
        while (len && (text[len] & 0xC0) == 0x80) len--;   /* whole UTF-8 characters */
    }
    size_t ct_cap = 2 + len + TAG_LEN;
    size_t body_len = HELLO_LEN + 2 + ct_cap;
    uint8_t *body = malloc(body_len);
    uint8_t *plain = malloc(2 + len);
    s_tts.rx = malloc(TTS_RX_CAP);
    s_tts.plain = malloc(TTS_PLAIN_CAP);
    bool ok = body && plain && s_tts.rx && s_tts.plain && (s_tts.sock = open_socket()) >= 0;
    if (ok) {
        new_hello(s_tts.hello);
        memcpy(body, s_tts.hello, HELLO_LEN);
        plain[0] = T_TEXT;
        plain[1] = (uint8_t)voice;
        memcpy(plain + 2, text, len);
        uint8_t n[12];
        nonce(s_tts.hello, 0, n);
        size_t ct_len = 0;
        ok = psa_aead_encrypt(s_key, PSA_ALG_GCM, n, sizeof(n), s_tts.hello, HELLO_LEN, plain, 2 + len,
                              body + HELLO_LEN + 2, ct_cap, &ct_len) == PSA_SUCCESS;
        body[HELLO_LEN] = ct_len & 0xff;
        body[HELLO_LEN + 1] = ct_len >> 8;
        char req[200];
        int h = snprintf(req, sizeof(req),
                         "POST %s HTTP/1.0\r\nHost: %s\r\nContent-Type: application/octet-stream\r\n"
                         "Content-Length: %u\r\n\r\n", CONFIG_MUSE_TTS_PATH, host_header(), (unsigned)body_len);
        ok = ok && send_on(s_tts.sock, req, h) && send_on(s_tts.sock, body, body_len);
    }
    free(body);
    free(plain);
    if (!ok) {
        ESP_LOGW(TAG, "speech: can't reach the server");
        muse_tts_close();
        return false;
    }
    int fl = fcntl(s_tts.sock, F_GETFL, 0);
    fcntl(s_tts.sock, F_SETFL, fl | O_NONBLOCK);
    s_tts.counter = SERVER_COUNTER;
    s_tts.t0 = esp_timer_get_time();
    return true;
}

/* Reads what the socket has; false once it's closed or failed. */
static bool tts_fill(void)
{
    if (s_tts.rx_len >= TTS_RX_CAP) {
        tts_fail("frame too big");
        return false;
    }
    int n = recv(s_tts.sock, s_tts.rx + s_tts.rx_len, TTS_RX_CAP - s_tts.rx_len, 0);
    if (n > 0) {
        s_tts.rx_len += n;
        return true;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return false;
    if (!s_tts.done) tts_fail("server closed early");
    return false;
}

static void tts_consume(size_t n)
{
    memmove(s_tts.rx, s_tts.rx + n, s_tts.rx_len - n);
    s_tts.rx_len -= n;
}

/* Opens the next frame into s_tts.plain. False if none is complete yet. */
static bool tts_next_frame(void)
{
    for (;;) {
        if (!s_tts.head) {
            uint8_t *end = s_tts.rx_len >= 4 ? memmem(s_tts.rx, s_tts.rx_len, "\r\n\r\n", 4) : NULL;
            if (end) {
                int status = 0;
                s_tts.rx[s_tts.rx_len < TTS_RX_CAP ? s_tts.rx_len : TTS_RX_CAP - 1] = 0;
                sscanf((const char *)s_tts.rx, "HTTP/1.%*d %d", &status);
                if (status != 200) {
                    ESP_LOGW(TAG, "speech: server answered %d", status);
                    tts_fail("refused");
                    return false;
                }
                tts_consume(end + 4 - s_tts.rx);
                s_tts.head = true;
                continue;
            }
        } else if (s_tts.rx_len >= 2) {
            size_t ct_len = s_tts.rx[0] | s_tts.rx[1] << 8;
            if (ct_len <= TAG_LEN || ct_len > TTS_PLAIN_CAP + TAG_LEN) {
                tts_fail("bad frame");
                return false;
            }
            if (s_tts.rx_len >= 2 + ct_len) {
                uint8_t n[12];
                nonce(s_tts.hello, s_tts.counter++, n);
                size_t plain_len = 0;
                if (psa_aead_decrypt(s_key, PSA_ALG_GCM, n, sizeof(n), s_tts.hello, HELLO_LEN, s_tts.rx + 2,
                                     ct_len, s_tts.plain, TTS_PLAIN_CAP, &plain_len) != PSA_SUCCESS
                    || plain_len < 1) {
                    tts_fail("frame didn't open");
                    return false;
                }
                tts_consume(2 + ct_len);
                s_tts.plain_len = plain_len;
                s_tts.plain_off = 1;
                return true;
            }
        }
        if (!tts_fill()) return false;
    }
}

size_t muse_tts_read(int16_t *pcm, size_t frames)
{
    size_t out = 0;
    frames &= ~(size_t)1;
    while (s_tts.sock >= 0 && out < frames && !s_tts.failed) {
        if (s_tts.plain_off < s_tts.plain_len && s_tts.plain[0] == T_AUDIO) {
            size_t n = (s_tts.plain_len - s_tts.plain_off) * 2;
            if (n > frames - out) n = frames - out;
            muse_adpcm_decode_block(&s_tts.adpcm, s_tts.plain + s_tts.plain_off, n, pcm + out);
            if (!s_tts.delivered) {
                ESP_LOGI(TAG, "speech after %.2fs", (esp_timer_get_time() - s_tts.t0) / 1e6);
            }
            s_tts.plain_off += n / 2;
            s_tts.delivered += n;
            out += n;
            continue;
        }
        s_tts.plain_off = s_tts.plain_len;
        if (s_tts.done || !tts_next_frame()) break;
        if (s_tts.plain[0] == T_SEGMENT && s_tts.plain_len >= 7) {
            const uint8_t *p = s_tts.plain + 1;
            s_tts.seg_from = s_tts.seg_to;
            s_tts.seg_samples = p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
            s_tts.seg_to = p[4] | p[5] << 8;
            s_tts.seg_base = s_tts.delivered;
        } else if (s_tts.plain[0] == T_END) {
            s_tts.done = true;
            ESP_LOGI(TAG, "speech: %.2fs in %.2fs", s_tts.delivered / (double)RATE,
                     (esp_timer_get_time() - s_tts.t0) / 1e6);
        }
    }
    return out;
}

int muse_tts_state(void)
{
    if (s_tts.failed) return -1;
    bool drained = s_tts.plain_off >= s_tts.plain_len;
    return s_tts.sock >= 0 && !(s_tts.done && drained) ? 0 : 1;
}

size_t muse_tts_text_at(void)
{
    if (!s_tts.seg_samples) return s_tts.seg_from;
    uint32_t in = s_tts.delivered - s_tts.seg_base;
    if (in > s_tts.seg_samples) in = s_tts.seg_samples;
    return s_tts.seg_from + (size_t)(s_tts.seg_to - s_tts.seg_from) * in / s_tts.seg_samples;
}

void muse_tts_close(void)
{
    if (s_tts.sock >= 0) close(s_tts.sock);
    free(s_tts.rx);
    free(s_tts.plain);
    memset(&s_tts, 0, sizeof(s_tts));
    s_tts.sock = -1;
}

#endif /* CONFIG_MUSE_ASR_ENABLE */
