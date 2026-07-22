"""Generate Raeen's verified NID->name table from a candidate dictionary.

The candidate source (shadPS4's aerolib.inl, GPL-2.0-or-later) is treated as
UNTRUSTED input: an entry is admitted only if OUR OWN implementation of the SCE
NID hash reproduces the NID from the name. A NID is SHA-1(name || salt), so a
name that hashes to its NID *is* the name -- a preimage is proof, not a claim.
That makes the dictionary's own accuracy irrelevant to our correctness.

Mirrors crates/raeen-firmware/src/dynlib/nid.rs (nid_of + encode_nid).
"""
import hashlib
import re
import sys

SALT = bytes([0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
              0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30])
ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"


def nid_of(name: str) -> int:
    d = hashlib.sha1(name.encode() + SALT).digest()
    return int.from_bytes(d[0:8], "little")


def encode_nid(nid: int) -> str:
    out = []
    acc = 0
    bits = 0
    for b in nid.to_bytes(8, "big"):
        acc = (acc << 8) | b
        bits += 8
        while bits >= 6:
            bits -= 6
            out.append(ALPHABET[(acc >> bits) & 0x3F])
    if bits > 0:
        out.append(ALPHABET[(acc << (6 - bits)) & 0x3F])
    return bytes(out).decode()


def main(src: str, dst: str) -> int:
    # NOTE: aerolib.inl wraps long entries across lines, e.g.
    #     STUB(
    #         "++LHBqoQ1cQ",
    #         _ZN3verylongmangledname)
    # A line-oriented regex silently drops 42,271 of 94,276 entries. Parse the
    # whole text with DOTALL instead, and assert the recovered count below.
    text = open(src, encoding="utf-8", errors="replace").read()
    pat = re.compile(r'STUB\(\s*"([^"]+)"\s*,\s*([^)]+?)\s*\)', re.S)
    seen: dict[int, str] = {}
    total = rejected = 0
    for m in pat.finditer(text):
        total += 1
        claimed, name = m.group(1), m.group(2).strip()
        nid = nid_of(name)
        # THE GATE: our hash must reproduce the claimed NID.
        if encode_nid(nid) != claimed:
            rejected += 1
            continue
        prev = seen.get(nid)
        if prev is not None and prev != name:
            # Two distinct names hashing to one NID would be a real SHA-1
            # collision; report loudly rather than silently pick a winner.
            print(f"COLLISION {encode_nid(nid)}: {prev!r} vs {name!r}", file=sys.stderr)
            continue
        seen[nid] = name

    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for nid in sorted(seen):
            f.write(f"{nid:016x} {seen[nid]}\n")

    print(f"candidates: {total}")
    print(f"rejected (hash mismatch): {rejected}")
    print(f"ADMITTED (hash-proven, unique): {len(seen)}")
    # Sanity vectors: must be present and correct.
    for probe in ("sceKernelGetGPI", "scePthreadMutexTimedlock", "sceKernelAllocateDirectMemory"):
        n = nid_of(probe)
        ok = seen.get(n) == probe
        print(f"  probe {probe:32s} {encode_nid(n)}  {'OK' if ok else 'ABSENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
