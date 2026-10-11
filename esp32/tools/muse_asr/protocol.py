"""Wire format between the gadget and muse_asr's server.

The gadget POSTs a chunked HTTP body made of:

  hello  16 bytes, in the clear: b"MSR1" + 8-byte random nonce prefix + u32 LE sample rate
  frames u16 LE length + AES-256-GCM(key, nonce, plaintext, aad=hello)
         nonce = prefix + u32 BE counter, from 0
         plaintext = type byte + data: 0x01 PCM16 LE mono, 0x02 end of speech

and the server answers 200 with one frame (counter 0x80000000) whose plaintext
is 0x03 + UTF-8 JSON {"text": ...}. The pre-shared key is the only credential:
a frame that doesn't open ends the request.

Speech for a reply goes the other way (POST /tts, HTTP/1.0, Content-Length):
the body is the hello and one frame 0x04 + voice byte + UTF-8 text; the server streams
frames back from counter 0x80000000 until it closes the connection:

  0x05 segment  u32 LE samples, u16 LE byte offset in the text where it ends
  0x01 audio    IMA ADPCM, 16 kHz mono, low nibble first, one state for the
                whole reply (as components/muse/muse_adpcm.c)
  0x02 end
"""

import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"MSR1"
HELLO_LEN = 16
TAG_LEN = 16
T_AUDIO, T_END, T_RESULT, T_TEXT, T_SEGMENT = 1, 2, 3, 4, 5
SERVER_COUNTER = 0x80000000
MAX_PLAIN = 8192


class ProtocolError(Exception):
    pass


def make_hello(prefix: bytes, rate: int) -> bytes:
    assert len(prefix) == 8
    return MAGIC + prefix + struct.pack("<I", rate)


def parse_hello(hello: bytes):
    if len(hello) != HELLO_LEN or hello[:4] != MAGIC:
        raise ProtocolError("bad hello")
    rate = struct.unpack("<I", hello[12:16])[0]
    if rate not in (8000, 16000, 24000):
        raise ProtocolError("bad rate")
    return hello[4:12], rate


def _nonce(prefix: bytes, counter: int) -> bytes:
    return prefix + struct.pack(">I", counter)


def seal(key: bytes, hello: bytes, counter: int, kind: int, data: bytes = b"") -> bytes:
    plain = bytes([kind]) + data
    if len(plain) > MAX_PLAIN:
        raise ProtocolError("frame too big")
    ct = AESGCM(key).encrypt(_nonce(hello[4:12], counter), plain, hello)
    return struct.pack("<H", len(ct)) + ct


def open_frame(key: bytes, hello: bytes, counter: int, ct: bytes):
    try:
        plain = AESGCM(key).decrypt(_nonce(hello[4:12], counter), ct, hello)
    except Exception as e:  # InvalidTag
        raise ProtocolError("frame failed to open") from e
    if not plain:
        raise ProtocolError("empty frame")
    return plain[0], plain[1:]


class FrameReader:
    """Splits a byte stream into frames: feed() bytes, then iterate frames()."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf += data

    def frames(self):
        while len(self.buf) >= 2:
            n = struct.unpack("<H", self.buf[:2])[0]
            if n < TAG_LEN + 1 or n > MAX_PLAIN + TAG_LEN:
                raise ProtocolError("bad frame length")
            if len(self.buf) < 2 + n:
                return
            ct = bytes(self.buf[2:2 + n])
            del self.buf[:2 + n]
            yield ct


# IMA ADPCM exactly as components/muse/muse_adpcm.c.
STEPS = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88,
    97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658,
    724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660,
    4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818,
    18500, 20350, 22385, 24623, 27086, 29794, 32767,
]
INDEX_STEP = [-1, -1, -1, -1, 2, 4, 6, 8]


class Adpcm:
    def __init__(self):
        self.pred = 0
        self.index = 0

    def _step(self, code):
        st = STEPS[self.index]
        diff = st >> 3
        if code & 4:
            diff += st
        if code & 2:
            diff += st >> 1
        if code & 1:
            diff += st >> 2
        pred = self.pred - diff if code & 8 else self.pred + diff
        self.pred = 32767 if pred > 32767 else (-32768 if pred < -32768 else pred)
        index = self.index + INDEX_STEP[code & 7]
        self.index = 0 if index < 0 else (88 if index > 88 else index)
        return self.pred

    def _encode(self, sample):
        diff = sample - self.pred
        code = 0
        if diff < 0:
            code = 8
            diff = -diff
        st = STEPS[self.index]
        if diff >= st:
            code |= 4
            diff -= st
        if diff >= st >> 1:
            code |= 2
            diff -= st >> 1
        if diff >= st >> 2:
            code |= 1
        self._step(code)
        return code

    def encode(self, samples):
        """Even count of int16 samples -> bytes, first of each pair in the low nibble."""
        out = bytearray(len(samples) // 2)
        for i in range(0, len(samples) - 1, 2):
            lo = self._encode(int(samples[i]))
            out[i // 2] = lo | self._encode(int(samples[i + 1])) << 4
        return bytes(out)

    def decode(self, data):
        pcm = []
        for b in data:
            pcm.append(self._step(b & 0x0F))
            pcm.append(self._step(b >> 4))
        return pcm
