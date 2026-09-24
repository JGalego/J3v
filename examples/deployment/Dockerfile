# A j3v server in a scratch image. The runtime is one static musl binary with no allocator surprises,
# no Python and no model server -- so the image is the binary, the encoder and the artifact, nothing else.
#
#   docker build -t j3v-triage --build-arg VERSION=v0.1.1 .
#   docker run --rm -p 8000:8000 j3v-triage
FROM alpine:3.20 AS fetch
ARG VERSION=v0.1.1
ARG TARGET=x86_64-unknown-linux-musl
RUN apk add --no-cache curl tar
WORKDIR /out
# The binary, checksum-verified against the release's SHA256SUMS.
RUN set -eu; \
    base="https://github.com/JGalego/J3v/releases/download/$VERSION"; \
    name="j3v-$VERSION-$TARGET"; \
    curl -fsSL "$base/$name.tar.gz" -o j3v.tar.gz; \
    curl -fsSL "$base/SHA256SUMS" -o SHA256SUMS; \
    grep " $name.tar.gz\$" SHA256SUMS | sed "s| $name.tar.gz| j3v.tar.gz|" | sha256sum -c -; \
    tar xzf j3v.tar.gz; mv "$name/j3v" /out/j3v; chmod +x /out/j3v
# The shared int8 encoder and the example artifact, from the same release.
RUN set -eu; \
    base="https://github.com/JGalego/J3v/releases/download/$VERSION"; \
    curl -fsSL "$base/minilm.j3a" -o minilm.j3a; \
    curl -fsSL "$base/support_triage.pi.j3a" -o model.j3a

FROM scratch
COPY --from=fetch /out/j3v /j3v
COPY --from=fetch /out/minilm.j3a /models/minilm.j3a
COPY --from=fetch /out/model.j3a /models/model.j3a
EXPOSE 8000
# --threads 1 matches the conformance runs and keeps latency predictable on a small box; raise it for throughput.
ENTRYPOINT ["/j3v", "serve", "/models/model.j3a", "--encoder", "/models/minilm.j3a", "--addr", "0.0.0.0:8000", "--threads", "1"]
