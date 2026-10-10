// Relay route: send Muse's hostnames through your own TCP relay, one relay
// port per hostname (each port forwards to that host's port 443, e.g. an Xray
// "tunnel"/dokodemo-door inbound). TLS, and its certificate check against the
// real hostname, still runs end to end. See Kconfig.
//
// The resolve hook hands entry i of RELAY_ROUTE_MAP the placeholder address
// 198.18.0.(i+1) (RFC 2544 benchmarking range, never routed); the connect
// wrapper turns placeholder:443 into RELAY_ROUTE_IP:<that entry's port>.

#include "sdkconfig.h"

#if CONFIG_RELAY_ROUTE_ENABLE

#if !CONFIG_LWIP_HOOK_NETCONN_EXT_RESOLVE_CUSTOM
#error "CONFIG_RELAY_ROUTE_ENABLE needs CONFIG_LWIP_HOOK_NETCONN_EXT_RESOLVE_CUSTOM=y"
#endif

#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "esp_log.h"
#include "lwip/api.h"
#include "lwip/ip_addr.h"
#include "lwip/sockets.h"

#define PLACEHOLDER_NET 0xC6120000u // 198.18.0.0
#define MAX_ENTRIES 8

static const char *TAG = "relay_route";

// Finds name in RELAY_ROUTE_MAP ("host=port host=port"). Returns the entry's
// index and sets *port, or returns -1.
static int lookup_name(const char *name, int *port)
{
    const char *p = CONFIG_RELAY_ROUTE_MAP;
    size_t n = strlen(name);
    for (int i = 0; *p && i < MAX_ENTRIES; i++) {
        while (*p == ' ' || *p == ',') {
            p++;
        }
        const char *eq = p;
        while (*eq && *eq != '=' && *eq != ' ' && *eq != ',') {
            eq++;
        }
        if (*eq != '=') {
            return -1;
        }
        int entry_port = atoi(eq + 1);
        if ((size_t)(eq - p) == n && strncasecmp(p, name, n) == 0 && entry_port > 0) {
            *port = entry_port;
            return i;
        }
        p = eq + 1;
        while (*p && *p != ' ' && *p != ',') {
            p++;
        }
    }
    return -1;
}

// Port of entry i, or 0.
static int lookup_index(int index)
{
    const char *p = CONFIG_RELAY_ROUTE_MAP;
    for (int i = 0; *p && i < MAX_ENTRIES; i++) {
        while (*p == ' ' || *p == ',') {
            p++;
        }
        const char *eq = strchr(p, '=');
        if (!eq) {
            return 0;
        }
        if (i == index) {
            return atoi(eq + 1);
        }
        p = eq + 1;
        while (*p && *p != ' ' && *p != ',') {
            p++;
        }
    }
    return 0;
}

int lwip_hook_netconn_external_resolve(const char *name, ip_addr_t *addr, u8_t addrtype, err_t *err)
{
    int port;
    int index = addrtype == NETCONN_DNS_IPV6 ? -1 : lookup_name(name, &port);
    if (index < 0) {
        return 0;
    }
    ip4_addr_t placeholder = { .addr = lwip_htonl(PLACEHOLDER_NET + index + 1) };
    ip_addr_copy_from_ip4(*addr, placeholder);
    *err = ERR_OK;
    ESP_LOGI(TAG, "%s -> relay %s:%d", name, CONFIG_RELAY_ROUTE_IP, port);
    return 1;
}

int __real_lwip_connect(int s, const struct sockaddr *name, socklen_t namelen);

int __wrap_lwip_connect(int s, const struct sockaddr *name, socklen_t namelen)
{
    if (name && name->sa_family == AF_INET && namelen >= sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)name;
        uint32_t host = lwip_ntohl(in->sin_addr.s_addr);
        if ((host & 0xFFFFFF00u) == PLACEHOLDER_NET) {
            int port = lookup_index((int)(host & 0xFF) - 1);
            ip4_addr_t relay;
            if (port <= 0 || in->sin_port != lwip_htons(443) || !ip4addr_aton(CONFIG_RELAY_ROUTE_IP, &relay)) {
                errno = EHOSTUNREACH;
                return -1;
            }
            struct sockaddr_in moved = *in;
            moved.sin_addr.s_addr = relay.addr;
            moved.sin_port = lwip_htons(port);
            return __real_lwip_connect(s, (const struct sockaddr *)&moved, sizeof(moved));
        }
    }
    return __real_lwip_connect(s, name, namelen);
}

#endif // CONFIG_RELAY_ROUTE_ENABLE
