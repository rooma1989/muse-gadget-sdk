"""Wire format between the gadget and muse_asr's server.

The gadget POSTs a chunked HTTP body made of:

  hello  16 bytes, in the clear: b"MSR1" + 8-byte random nonce prefix + u32 LE sample rate
  frames u16 LE length + AES-256-GCM(key, nonce, plaintext, aad=hello)
         nonce = prefix + u32 BE counter, from 0
         plaintext = type byte + data: 0x01 PCM16 LE mono, 0x02 end of speech

and the server answers 200 with one frame (counter 0x80000000) whose plaintext
is 0x03 + UTF-8 JSON {"text": ...}. The pre-shared key is the only credential:
a frame that doesn't open ends the request.
"""

import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"MSR1"
HELLO_LEN = 16
TAG_LEN = 16
T_AUDIO, T_END, T_RESULT = 1, 2, 3
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
