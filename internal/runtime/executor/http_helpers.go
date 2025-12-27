package executor

import (
	"compress/flate"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/andybalholm/brotli"
	"github.com/klauspost/compress/zstd"
)

// =============================================================================
// HTTP Header Helpers
// =============================================================================

// HeaderConfig configures HTTP headers for API requests.
// It provides a unified way to apply common headers across different providers.
type HeaderConfig struct {
	// Token is the Bearer token value for Authorization header.
	Token string

	// UserAgent is the User-Agent header value (optional).
	UserAgent string

	// ExtraHeaders are provider-specific headers to set.
	ExtraHeaders map[string]string

	// StreamHeaders are additional headers for streaming requests (e.g., Cache-Control).
	StreamHeaders map[string]string
}

// ApplyAPIHeaders applies common HTTP headers for OpenAI-compatible APIs.
// It sets Content-Type, Authorization (Bearer token), Accept, and optional User-Agent.
// The stream parameter controls Accept header (text/event-stream vs application/json).
func ApplyAPIHeaders(r *http.Request, cfg HeaderConfig, stream bool) {
	// Standard headers (always set)
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+cfg.Token)

	// Accept header based on streaming mode
	if stream {
		r.Header.Set("Accept", "text/event-stream")
		// Apply stream-specific headers
		for k, v := range cfg.StreamHeaders {
			r.Header.Set(k, v)
		}
	} else {
		r.Header.Set("Accept", "application/json")
	}

	// User-Agent (if provided)
	if cfg.UserAgent != "" {
		r.Header.Set("User-Agent", cfg.UserAgent)
	}

	// Extra provider-specific headers
	for k, v := range cfg.ExtraHeaders {
		r.Header.Set(k, v)
	}
}

// =============================================================================
// Decompression Pools
// =============================================================================

// GzipReaderPool reduces allocations for gzip decompression.
// gzip.Reader can be reset and reused across requests.
var GzipReaderPool = sync.Pool{
	New: func() any {
		// Return nil; will be initialized on first use with Reset()
		return new(gzip.Reader)
	},
}

// zstdDecoderPool reduces allocations for zstd decompression.
// zstd.Decoder is expensive to create, pooling is beneficial.
var zstdDecoderPool = sync.Pool{
	New: func() any {
		// Create with default options; will be reset on Get
		decoder, _ := zstd.NewReader(nil)
		return decoder
	},
}

// brotliReaderPool reduces allocations for brotli decompression.
// brotli.Reader can be reset and reused across requests.
var brotliReaderPool = sync.Pool{
	New: func() any {
		// Return nil; will be initialized on first use with Reset()
		return new(brotli.Reader)
	},
}

// compositeReadCloser wraps a reader with multiple closers that need to be called.
// This is used for decompression readers where both the decompressor and the
// underlying body need to be closed properly.
type compositeReadCloser struct {
	io.Reader
	closers []func() error
}

func (c *compositeReadCloser) Close() error {
	var firstErr error
	for i := range c.closers {
		if c.closers[i] == nil {
			continue
		}
		if err := c.closers[i](); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// pooledGzipReadCloser wraps a pooled gzip.Reader with proper cleanup.
type pooledGzipReadCloser struct {
	gr   *gzip.Reader
	body io.ReadCloser
}

func (p *pooledGzipReadCloser) Read(b []byte) (int, error) {
	return p.gr.Read(b)
}

func (p *pooledGzipReadCloser) Close() error {
	// Close the gzip reader and return it to the pool
	err := p.gr.Close()
	GzipReaderPool.Put(p.gr)
	// Close the underlying body
	if bodyErr := p.body.Close(); bodyErr != nil && err == nil {
		err = bodyErr
	}
	return err
}

// pooledZstdReadCloser wraps a pooled zstd.Decoder with proper cleanup.
type pooledZstdReadCloser struct {
	decoder *zstd.Decoder
	body    io.ReadCloser
}

func (p *pooledZstdReadCloser) Read(b []byte) (int, error) {
	return p.decoder.Read(b)
}

func (p *pooledZstdReadCloser) Close() error {
	// Reset decoder to release resources, then return to pool
	p.decoder.Reset(nil)
	zstdDecoderPool.Put(p.decoder)
	// Close the underlying body
	return p.body.Close()
}

// pooledBrotliReadCloser wraps a pooled brotli.Reader with proper cleanup.
type pooledBrotliReadCloser struct {
	br   *brotli.Reader
	body io.ReadCloser
}

func (p *pooledBrotliReadCloser) Read(b []byte) (int, error) {
	return p.br.Read(b)
}

func (p *pooledBrotliReadCloser) Close() error {
	// Discard remaining data to allow reuse, then return to pool
	io.Copy(io.Discard, p.br)
	brotliReaderPool.Put(p.br)
	// Close the underlying body
	return p.body.Close()
}

// decodeResponseBody wraps the response body with the appropriate decompression reader
// based on the Content-Encoding header. Supports gzip, deflate, br (brotli), and zstd.
// Returns the original body if no encoding is specified or if encoding is "identity".
// Uses pooled readers for gzip and zstd to reduce allocations.
func decodeResponseBody(body io.ReadCloser, contentEncoding string) (io.ReadCloser, error) {
	if body == nil {
		return nil, fmt.Errorf("response body is nil")
	}
	if contentEncoding == "" {
		return body, nil
	}
	encodings := strings.Split(contentEncoding, ",")
	for _, raw := range encodings {
		encoding := strings.TrimSpace(strings.ToLower(raw))
		switch encoding {
		case "", "identity":
			continue
		case "gzip":
			// Get pooled gzip reader
			gr := GzipReaderPool.Get().(*gzip.Reader)
			if err := gr.Reset(body); err != nil {
				GzipReaderPool.Put(gr)
				_ = body.Close()
				return nil, fmt.Errorf("failed to reset gzip reader: %w", err)
			}
			return &pooledGzipReadCloser{gr: gr, body: body}, nil
		case "deflate":
			deflateReader := flate.NewReader(body)
			return &compositeReadCloser{
				Reader: deflateReader,
				closers: []func() error{
					deflateReader.Close,
					func() error { return body.Close() },
				},
			}, nil
		case "br":
			// Get pooled brotli reader
			br := brotliReaderPool.Get().(*brotli.Reader)
			if err := br.Reset(body); err != nil {
				brotliReaderPool.Put(br)
				_ = body.Close()
				return nil, fmt.Errorf("failed to reset brotli reader: %w", err)
			}
			return &pooledBrotliReadCloser{br: br, body: body}, nil
		case "zstd":
			// Get pooled zstd decoder
			decoder := zstdDecoderPool.Get().(*zstd.Decoder)
			if err := decoder.Reset(body); err != nil {
				zstdDecoderPool.Put(decoder)
				_ = body.Close()
				return nil, fmt.Errorf("failed to reset zstd decoder: %w", err)
			}
			return &pooledZstdReadCloser{decoder: decoder, body: body}, nil
		default:
			continue
		}
	}
	return body, nil
}
