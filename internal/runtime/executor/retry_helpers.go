// Package executor provides retry logic for multi-target execution with fallback.
//
// Retry Handling:
//
// This file provides two layers of retry logic:
//
//  1. RetryHandler (new): Modern abstraction for multi-target execution
//     - Configurable retry status codes (default: 429)
//     - Configurable fallback codes (default: 429, 503)
//     - Exponential backoff with server-provided delay hints
//     - Multi-target fallback support
//
//  2. rateLimitRetrier (legacy): Backward-compatible retry for rate limits
//     - Used by older executor implementations
//     - 1 retry with exponential backoff
//
// RetryAction flow:
//
//	RetryActionSuccess      -> Request succeeded, return response
//	RetryActionRetryCurrent -> Wait and retry same target
//	RetryActionContinueNext -> Skip to next target in pool
//	RetryActionFail         -> All retries exhausted, return error
//
// Usage:
//
//	handler := NewRetryHandler(DefaultRetryConfig())
//	action, err := handler.HandleResponse(ctx, resp.StatusCode, body, hasNextTarget)
//	switch action {
//	case RetryActionSuccess:
//	    return resp, nil
//	case RetryActionRetryCurrent:
//	    continue // retry loop
//	case RetryActionContinueNext:
//	    break // outer loop to next target
//	case RetryActionFail:
//	    return nil, err
//	}
package executor

import (
	"context"
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

const (
	// Rate limit retry settings: 1 retry to handle transient glitches, then failover to next provider
	rateLimitMaxRetries = 1
)

// RetryConfig defines retry behavior for multi-target execution
type RetryConfig struct {
	MaxRetries       int           // Max retries per target (default: 1)
	BaseDelay        time.Duration // Exponential backoff base (default: 1s)
	MaxDelay         time.Duration // Cap per-retry delay (default: 20s)
	RetryStatusCodes []int         // Status codes to retry (default: 429)
	FallbackCodes    []int         // Codes to fallback to next target (default: 429, 503)
	RetryOnErrors    bool          // Retry on network errors
}

// RetryHandler manages retry logic with multi-target fallback
type RetryHandler struct {
	config  RetryConfig
	retrier rateLimitRetrier // Reuse existing implementation
}

// RetryAction represents the next action to take
type RetryAction int

const (
	RetryActionSuccess      RetryAction = iota // Request succeeded
	RetryActionContinueNext                    // Fallback to next target
	RetryActionRetryCurrent                    // Retry current target after delay
	RetryActionFail                            // All retries exhausted, fail
)

// String returns a string representation of the RetryAction
func (a RetryAction) String() string {
	switch a {
	case RetryActionSuccess:
		return "Success"
	case RetryActionContinueNext:
		return "ContinueNext"
	case RetryActionRetryCurrent:
		return "RetryCurrent"
	case RetryActionFail:
		return "Fail"
	default:
		return fmt.Sprintf("Unknown(%d)", int(a))
	}
}

// DefaultRetryConfig returns sensible defaults for retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:       1,
		BaseDelay:        RateLimitBaseDelay,
		MaxDelay:         RateLimitMaxDelay,
		RetryStatusCodes: []int{429},
		FallbackCodes:    []int{429, 503},
		RetryOnErrors:    true,
	}
}

// AntigravityRetryConfig returns config matching Antigravity behavior
// Antigravity uses more aggressive retry with longer delays for quota limits
func AntigravityRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:       2,
		BaseDelay:        AntigravityRetryBaseDelay,
		MaxDelay:         AntigravityRetryMaxDelay,
		RetryStatusCodes: []int{429, 503, 500},
		FallbackCodes:    []int{429, 503},
		RetryOnErrors:    true,
	}
}

// NewRetryHandler creates a new RetryHandler with the given config
func NewRetryHandler(cfg RetryConfig) *RetryHandler {
	// Apply defaults for zero values
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 1
	}
	if cfg.BaseDelay == 0 {
		cfg.BaseDelay = RateLimitBaseDelay
	}
	if cfg.MaxDelay == 0 {
		cfg.MaxDelay = RateLimitMaxDelay
	}
	if len(cfg.RetryStatusCodes) == 0 {
		cfg.RetryStatusCodes = []int{429}
	}
	if len(cfg.FallbackCodes) == 0 {
		cfg.FallbackCodes = []int{429, 503}
	}

	return &RetryHandler{
		config:  cfg,
		retrier: rateLimitRetrier{},
	}
}

// HandleResponse evaluates an HTTP response and returns the appropriate action.
// It considers status codes, retry configuration, and whether a fallback target is available.
func (h *RetryHandler) HandleResponse(ctx context.Context, statusCode int, body []byte, hasNextTarget bool) (RetryAction, error) {
	// Success case (2xx)
	if statusCode >= 200 && statusCode < 300 {
		return RetryActionSuccess, nil
	}

	// Check if this is a retryable status code
	isRetryable := h.isRetryableStatus(statusCode)
	isFallbackCode := h.isFallbackCode(statusCode)

	// If it's a fallback code and we have more targets, continue to next
	if isFallbackCode && hasNextTarget {
		log.Debugf("retry_handler: status %d, falling back to next target", statusCode)
		return RetryActionContinueNext, nil
	}

	// If it's retryable and we haven't exhausted retries
	if isRetryable && h.retrier.retryCount < h.config.MaxRetries {
		delay := h.calculateDelay(body)
		h.retrier.retryCount++
		log.Debugf("retry_handler: status %d, waiting %v before retry %d/%d",
			statusCode, delay, h.retrier.retryCount, h.config.MaxRetries)

		select {
		case <-ctx.Done():
			return RetryActionFail, ctx.Err()
		case <-time.After(delay):
		}

		return RetryActionRetryCurrent, nil
	}

	// If we can fallback to next target (even if not a typical fallback code)
	if hasNextTarget && isRetryable {
		log.Debugf("retry_handler: retries exhausted for status %d, trying next target", statusCode)
		return RetryActionContinueNext, nil
	}

	// All options exhausted
	log.Debugf("retry_handler: status %d, no more retries or targets available", statusCode)
	return RetryActionFail, nil
}

// HandleError evaluates a network/transport error and returns the appropriate action.
func (h *RetryHandler) HandleError(ctx context.Context, err error, hasNextTarget bool) (RetryAction, error) {
	if err == nil {
		return RetryActionSuccess, nil
	}

	// Check context cancellation first
	if ctx.Err() != nil {
		return RetryActionFail, ctx.Err()
	}

	// If we have more targets, try next one
	if hasNextTarget {
		log.Debugf("retry_handler: error occurred, falling back to next target: %v", err)
		return RetryActionContinueNext, nil
	}

	// If configured to retry on errors and haven't exhausted retries
	if h.config.RetryOnErrors && h.retrier.retryCount < h.config.MaxRetries {
		delay := h.calculateDelayForError()
		h.retrier.retryCount++
		log.Debugf("retry_handler: error occurred, waiting %v before retry %d/%d: %v",
			delay, h.retrier.retryCount, h.config.MaxRetries, err)

		select {
		case <-ctx.Done():
			return RetryActionFail, ctx.Err()
		case <-time.After(delay):
		}

		return RetryActionRetryCurrent, nil
	}

	return RetryActionFail, err
}

// Reset resets the retry counter for a new request cycle
func (h *RetryHandler) Reset() {
	h.retrier.retryCount = 0
}

// RetryCount returns the current retry count
func (h *RetryHandler) RetryCount() int {
	return h.retrier.retryCount
}

// Config returns the current retry configuration
func (h *RetryHandler) Config() RetryConfig {
	return h.config
}

// isRetryableStatus checks if the status code should trigger a retry
func (h *RetryHandler) isRetryableStatus(statusCode int) bool {
	for _, code := range h.config.RetryStatusCodes {
		if statusCode == code {
			return true
		}
	}
	return false
}

// isFallbackCode checks if the status code should trigger a fallback to next target
func (h *RetryHandler) isFallbackCode(statusCode int) bool {
	for _, code := range h.config.FallbackCodes {
		if statusCode == code {
			return true
		}
	}
	return false
}

// calculateDelay calculates delay using server hints or exponential backoff
func (h *RetryHandler) calculateDelay(body []byte) time.Duration {
	// Try to use server-provided retry delay
	if serverDelay, err := parseRetryDelay(body); err == nil && serverDelay != nil {
		delay := *serverDelay + 500*time.Millisecond // Add buffer
		if delay > h.config.MaxDelay {
			delay = h.config.MaxDelay
		}
		return delay
	}

	// Fall back to exponential backoff
	return h.calculateDelayForError()
}

// calculateDelayForError calculates exponential backoff delay for errors
func (h *RetryHandler) calculateDelayForError() time.Duration {
	delay := h.config.BaseDelay * time.Duration(1<<h.retrier.retryCount)
	if delay > h.config.MaxDelay {
		delay = h.config.MaxDelay
	}
	return delay
}

// rateLimitRetrier handles rate limit (429) errors with exponential backoff retry logic.
type rateLimitRetrier struct {
	retryCount int
}

// rateLimitAction represents the action to take after handling a rate limit error.
type rateLimitAction int

const (
	rateLimitActionContinue    rateLimitAction = iota // Continue to next model
	rateLimitActionRetry                              // Retry same model after delay
	rateLimitActionMaxExceeded                        // Max retries exceeded, stop
)

// handleRateLimit processes a 429 rate limit error and returns the appropriate action.
// It handles model fallback first, then applies exponential backoff with retries.
// Returns the action to take and waits if necessary (respecting context cancellation).
func (r *rateLimitRetrier) handleRateLimit(ctx context.Context, hasNextModel bool, errorBody []byte) (rateLimitAction, error) {
	// Try next model first if available
	if hasNextModel {
		return rateLimitActionContinue, nil
	}

	// No more models - apply exponential backoff with retries
	if r.retryCount >= rateLimitMaxRetries {
		log.Debug("executor: rate limited, max retries exceeded")
		return rateLimitActionMaxExceeded, nil
	}

	delay := r.calculateDelay(errorBody)
	r.retryCount++
	log.Debugf("executor: rate limited, waiting %v before retry %d/%d", delay, r.retryCount, rateLimitMaxRetries)

	select {
	case <-ctx.Done():
		return rateLimitActionMaxExceeded, ctx.Err()
	case <-time.After(delay):
	}

	return rateLimitActionRetry, nil
}

// calculateDelay calculates the delay for rate limit retry with exponential backoff.
// It first tries to use the server-provided retry delay from the error response,
// then falls back to exponential backoff: 1s, 2s, 4s, 8s, 16s (capped at 20s).
func (r *rateLimitRetrier) calculateDelay(errorBody []byte) time.Duration {
	// First, try to use server-provided retry delay
	if serverDelay, err := parseRetryDelay(errorBody); err == nil && serverDelay != nil {
		delay := *serverDelay
		// Add a small buffer to the server-provided delay
		delay += 500 * time.Millisecond
		if delay > RateLimitMaxDelay {
			delay = RateLimitMaxDelay
		}
		return delay
	}

	// Fall back to exponential backoff: baseDelay * 2^retryCount
	delay := RateLimitBaseDelay * time.Duration(1<<r.retryCount)
	if delay > RateLimitMaxDelay {
		delay = RateLimitMaxDelay
	}
	return delay
}

// parseRetryDelay extracts the retry delay from a Google API 429 error response.
// The error response contains a RetryInfo.retryDelay field in the format "0.847655010s".
// Handles both formats:
//   - Object: {"error": {"details": [...]}}
//   - Array:  [{"error": {"details": [...]}}]
//
// Returns the parsed duration or an error if it cannot be determined.
func parseRetryDelay(errorBody []byte) (*time.Duration, error) {
	// Try multiple paths to handle different response formats
	paths := []string{
		"error.details",   // Standard: {"error": {"details": [...]}}
		"0.error.details", // Array wrapped: [{"error": {"details": [...]}]
	}

	var details gjson.Result
	for _, path := range paths {
		details = gjson.GetBytes(errorBody, path)
		if details.Exists() && details.IsArray() {
			break
		}
	}

	if !details.Exists() || !details.IsArray() {
		return nil, fmt.Errorf("no error.details found")
	}

	for _, detail := range details.Array() {
		typeVal := detail.Get("@type").String()
		if typeVal == "type.googleapis.com/google.rpc.RetryInfo" {
			retryDelay := detail.Get("retryDelay").String()
			if retryDelay != "" {
				// Parse duration string like "0.847655010s"
				duration, err := time.ParseDuration(retryDelay)
				if err != nil {
					return nil, fmt.Errorf("failed to parse duration: %w", err)
				}
				return &duration, nil
			}
		}
	}

	return nil, fmt.Errorf("no RetryInfo found")
}

// ParseQuotaRetryDelay extracts the full quota reset delay from a Google API 429 error response.
// Unlike parseRetryDelay which is used for short-term retries (capped at 20s), this function
// returns the actual quota reset time which can be hours.
// It checks multiple sources in order of preference:
//  1. RetryInfo.retryDelay (e.g., "7118.204539195s") - most accurate
//  2. ErrorInfo.metadata.quotaResetDelay (e.g., "1h58m38.204539195s") - human-readable format
//
// Returns nil if no quota delay information is found.
func ParseQuotaRetryDelay(errorBody []byte) *time.Duration {
	paths := []string{
		"error.details",
		"0.error.details",
	}

	var details gjson.Result
	for _, path := range paths {
		details = gjson.GetBytes(errorBody, path)
		if details.Exists() && details.IsArray() {
			break
		}
	}

	if !details.Exists() || !details.IsArray() {
		return nil
	}

	var quotaResetDelay *time.Duration

	for _, detail := range details.Array() {
		typeVal := detail.Get("@type").String()

		// Prefer RetryInfo.retryDelay (more precise, in seconds)
		if typeVal == "type.googleapis.com/google.rpc.RetryInfo" {
			retryDelay := detail.Get("retryDelay").String()
			if retryDelay != "" {
				if duration, err := time.ParseDuration(retryDelay); err == nil && duration > 0 {
					return &duration
				}
			}
		}

		// Fallback to ErrorInfo.metadata.quotaResetDelay
		if typeVal == "type.googleapis.com/google.rpc.ErrorInfo" {
			quotaDelay := detail.Get("metadata.quotaResetDelay").String()
			if quotaDelay != "" {
				if duration, err := time.ParseDuration(quotaDelay); err == nil && duration > 0 {
					quotaResetDelay = &duration
				}
			}
		}
	}

	return quotaResetDelay
}
