// Package executor provides HTTP transport configuration for high-performance scenarios.
package executor

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"net/url"
	"time"
)

// TransportConfig contains tuned settings for high-performance HTTP transport.
// These values are optimized for API gateway workloads with many concurrent connections.
var TransportConfig = struct {
	MaxIdleConns          int
	MaxIdleConnsPerHost   int
	MaxConnsPerHost       int
	IdleConnTimeout       time.Duration
	TLSHandshakeTimeout   time.Duration
	ExpectContinueTimeout time.Duration
	DialTimeout           time.Duration
	KeepAlive             time.Duration
}{
	MaxIdleConns:          1000,
	MaxIdleConnsPerHost:   100, // Default is 2, too low for API gateways
	MaxConnsPerHost:       200,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: 1 * time.Second,
	DialTimeout:           30 * time.Second,
	KeepAlive:             30 * time.Second,
}

// SharedTransport is the default HTTP transport for direct connections.
// Used when no proxy is configured and no context RoundTripper is provided.
var SharedTransport = &http.Transport{
	MaxIdleConns:          TransportConfig.MaxIdleConns,
	MaxIdleConnsPerHost:   TransportConfig.MaxIdleConnsPerHost,
	MaxConnsPerHost:       TransportConfig.MaxConnsPerHost,
	IdleConnTimeout:       TransportConfig.IdleConnTimeout,
	TLSHandshakeTimeout:   TransportConfig.TLSHandshakeTimeout,
	ExpectContinueTimeout: TransportConfig.ExpectContinueTimeout,
	ResponseHeaderTimeout: 60 * time.Second,
	ForceAttemptHTTP2:     true,
	DisableCompression:    false,
	DialContext: (&net.Dialer{
		Timeout:   TransportConfig.DialTimeout,
		KeepAlive: TransportConfig.KeepAlive,
	}).DialContext,
	TLSClientConfig: &tls.Config{
		MinVersion: tls.VersionTLS12,
	},
}

// ProxyTransport creates an HTTP transport with proxy configuration.
func ProxyTransport(proxyURL *url.URL) *http.Transport {
	return &http.Transport{
		Proxy:                 http.ProxyURL(proxyURL),
		MaxIdleConns:          TransportConfig.MaxIdleConns,
		MaxIdleConnsPerHost:   TransportConfig.MaxIdleConnsPerHost,
		MaxConnsPerHost:       TransportConfig.MaxConnsPerHost,
		IdleConnTimeout:       TransportConfig.IdleConnTimeout,
		TLSHandshakeTimeout:   TransportConfig.TLSHandshakeTimeout,
		ExpectContinueTimeout: TransportConfig.ExpectContinueTimeout,
		ResponseHeaderTimeout: 60 * time.Second,
		ForceAttemptHTTP2:     true,
		DisableCompression:    false,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}
}

// SOCKS5Transport creates an HTTP transport with SOCKS5 dialer.
func SOCKS5Transport(dialFunc func(network, addr string) (net.Conn, error)) *http.Transport {
	return &http.Transport{
		DialContext: func(_ context.Context, network, addr string) (net.Conn, error) {
			return dialFunc(network, addr)
		},
		MaxIdleConns:          TransportConfig.MaxIdleConns,
		MaxIdleConnsPerHost:   TransportConfig.MaxIdleConnsPerHost,
		MaxConnsPerHost:       TransportConfig.MaxConnsPerHost,
		IdleConnTimeout:       TransportConfig.IdleConnTimeout,
		TLSHandshakeTimeout:   TransportConfig.TLSHandshakeTimeout,
		ExpectContinueTimeout: TransportConfig.ExpectContinueTimeout,
		ResponseHeaderTimeout: 60 * time.Second,
		ForceAttemptHTTP2:     true,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}
}
