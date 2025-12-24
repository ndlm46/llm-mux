// Package executor provides common functionality for provider executors.
//
// BaseExecutor contains shared fields and methods that all concrete executors
// can embed to reduce boilerplate. Embedding is optional but recommended.
package executor

import (
	"context"
	"net/http"
	"time"

	"github.com/nghyane/llm-mux/internal/config"
	cliproxyauth "github.com/nghyane/llm-mux/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/nghyane/llm-mux/sdk/cliproxy/executor"
)

// BaseExecutor provides common functionality that all provider executors can embed.
// It contains the shared configuration and provides default implementations for
// commonly used methods across all executors.
//
// Usage:
//
//	type MyExecutor struct {
//	    BaseExecutor
//	}
//
//	func NewMyExecutor(cfg *config.Config) *MyExecutor {
//	    return &MyExecutor{BaseExecutor: BaseExecutor{Cfg: cfg}}
//	}
type BaseExecutor struct {
	Cfg *config.Config
}

func (b *BaseExecutor) Config() *config.Config {
	return b.Cfg
}

// PrepareRequest prepares the HTTP request for execution.
// This default implementation is a no-op that returns nil.
//
// This method exists in the ProviderExecutor interface to support:
//   - Backward compatibility: SDK users may have external executors that override this method
//   - Future extensibility: Allows injecting credentials/headers into raw HTTP requests
//     before execution (see sdk/cliproxy/auth/manager.go:RequestPreparer interface)
//   - Custom providers: External providers using the SDK can override this to add
//     custom authentication or request modification logic
//
// Note: All internal executors currently return nil (no-op) because they handle
// request preparation inline within Execute/ExecuteStream methods. The interface
// method is retained for SDK API stability and custom provider support.
//
// See also: sdk/cliproxy/auth/manager.go:InjectCredentials which calls this method
// through the RequestPreparer interface for SDK consumers.
func (b *BaseExecutor) PrepareRequest(_ *http.Request, _ *cliproxyauth.Auth) error {
	return nil
}

// NewHTTPClient creates an HTTP client with proper proxy configuration.
func (b *BaseExecutor) NewHTTPClient(ctx context.Context, auth *cliproxyauth.Auth, timeout time.Duration) *http.Client {
	return newProxyAwareHTTPClient(ctx, b.Cfg, auth, timeout)
}

// NewUsageReporter creates a new usage reporter for tracking API usage.
func (b *BaseExecutor) NewUsageReporter(ctx context.Context, provider, model string, auth *cliproxyauth.Auth) *usageReporter {
	return newUsageReporter(ctx, provider, model, auth)
}

// ApplyPayloadConfig applies payload default and override rules from configuration.
func (b *BaseExecutor) ApplyPayloadConfig(model string, payload []byte) []byte {
	return applyPayloadConfig(b.Cfg, model, payload)
}

// RefreshNoOp is a no-op refresh for executors that don't require token refresh.
func (b *BaseExecutor) RefreshNoOp(_ context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	return auth, nil
}

// CountTokensNotSupported returns a NotImplemented error for executors that don't support token counting.
func (b *BaseExecutor) CountTokensNotSupported(provider string) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, NewNotImplementedError("count tokens not supported for " + provider)
}
