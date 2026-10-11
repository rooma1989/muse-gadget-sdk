#!/usr/bin/env python3
"""Sends a WAV file through muse_asr the way the gadget does; prints the text.

  client.py KEY_FILE URL file.wav     e.g. http://203.0.113.10/TV/asr

The WAV must be 16-bit mono. Audio goes up in 100 ms frames, paced like a live
mic unless --fast.
"""

import argparse
import http.client
import json
import os
import struct
import sys
import time
import urllib.parse
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402


def body(key, hello, pcm, rate, fast):
    yield hello
    step = rate // 10 * 2
    counter = 0
    for off in range(0, len(pcm), step):
        yield P.seal(key, hello, counter, P.T_AUDIO, pcm[off:off + step])
        counter += 1
        if not fast:
            time.sleep(0.1)
    yield P.seal(key, hello, counter, P.T_END)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key_file")
    ap.add_argument("url")
    ap.add_argument("wav")
    ap.add_argument("--host-header", help="Host header (default: the URL's host)")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    key = bytes.fromhex(open(a.key_file).read().strip())
    with wave.open(a.wav) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "16-bit mono WAV"
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    hello = P.make_hello(os.urandom(8), rate)
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    t0 = time.monotonic()
    conn.request("POST", u.path, body=body(key, hello, pcm, rate, a.fast),
                 headers={"Host": a.host_header or u.netloc, "Content-Type": "application/octet-stream"},
                 encode_chunked=True)
    r = conn.getresponse()
    data = r.read()
    t1 = time.monotonic()
    if r.status != 200:
        sys.exit(f"HTTP {r.status}: {data[:200]!r}")
    n = struct.unpack("<H", data[:2])[0]
    kind, payload = P.open_frame(key, hello, P.SERVER_COUNTER, data[2:2 + n])
    assert kind == P.T_RESULT
    print(json.loads(payload)["text"])
    print(f"({len(pcm) / 2 / rate:.2f}s audio, {(t1 - t0) * 1000:.0f} ms total)", file=sys.stderr)


if __name__ == "__main__":
    main()
