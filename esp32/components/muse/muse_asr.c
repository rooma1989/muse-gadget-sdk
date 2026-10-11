/*
 * Push-to-talk through your own speech-to-text server. See muse_asr.h, and
 * tools/muse_asr/protocol.py for the wire format: a chunked HTTP POST of a
 * 16-byte hello and AES-256-GCM frames of PCM, answered by one sealed frame
 * holding {"text": ...}. Plain TCP to a nearby server: no second TLS session,
 * which a board without PSRAM has no room for beside Link's.
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

static void nonce(uint32_t counter, uint8_t out[12])
{
    memcpy(out, s_hello + 4, 8);
    out[8] = counter >> 24;
    out[9] = counter >> 16;
    out[10] = counter >> 8;
    out[11] = counter;
}

static bool send_all(const void *data, size_t len)
{
    const uint8_t *p = data;
    while (len) {
        int n = send(s_sock, p, len, 0);
        if (n <= 0) {
            ESP_LOGW(TAG, "send failed: errno %d", errno);
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
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
    nonce(s_counter++, n);
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

static bool connect_to_server(void)
{
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(CONFIG_MUSE_ASR_PORT) };
    if (inet_aton(CONFIG_MUSE_ASR_IP, &addr.sin_addr) == 0) {
        ESP_LOGE(TAG, "bad CONFIG_MUSE_ASR_IP");
        return false;
    }
    s_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s_sock < 0) return false;
    int fl = fcntl(s_sock, F_GETFL, 0);
    fcntl(s_sock, F_SETFL, fl | O_NONBLOCK);
    int rc = connect(s_sock, (struct sockaddr *)&addr, sizeof(addr));
    if (rc < 0 && errno == EINPROGRESS) {
        fd_set w;
        FD_ZERO(&w);
        FD_SET(s_sock, &w);
        struct timeval tv = { CONNECT_MS / 1000, (CONNECT_MS % 1000) * 1000 };
        int err = 0;
        socklen_t el = sizeof(err);
        rc = select(s_sock + 1, NULL, &w, NULL, &tv) == 1
             && getsockopt(s_sock, SOL_SOCKET, SO_ERROR, &err, &el) == 0 && err == 0 ? 0 : -1;
    }
    if (rc < 0) {
        ESP_LOGW(TAG, "can't connect to %s:%d", CONFIG_MUSE_ASR_IP, CONFIG_MUSE_ASR_PORT);
        return false;
    }
    fcntl(s_sock, F_SETFL, fl & ~O_NONBLOCK);
    struct timeval tv = { SEND_MS / 1000, (SEND_MS % 1000) * 1000 };
    setsockopt(s_sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int one = 1;
    setsockopt(s_sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return true;
}

bool muse_asr_begin(void)
{
    muse_asr_close();
    if (!load_key()) return false;
    s_plain = malloc(1 + FRAME_PCM);
    s_out = malloc(OUT_CAP);
    if (!s_plain || !s_out || !connect_to_server()) {
        muse_asr_close();
        return false;
    }
    char req[200];
    int n = snprintf(req, sizeof(req),
                     "POST %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/octet-stream\r\n"
                     "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                     CONFIG_MUSE_ASR_PATH, CONFIG_MUSE_ASR_HOST[0] ? CONFIG_MUSE_ASR_HOST : CONFIG_MUSE_ASR_IP);
    memcpy(s_hello, "MSR1", 4);
    esp_fill_random(s_hello + 4, 8);
    s_hello[12] = RATE & 0xff;
    s_hello[13] = (RATE >> 8) & 0xff;
    s_hello[14] = 0;
    s_hello[15] = 0;
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
    nonce(SERVER_COUNTER, n);
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

#endif /* CONFIG_MUSE_ASR_ENABLE */
