// Package executor provides common utilities for executor implementations.
//
// Error Handling:
//
// This file provides standardized error handling across all executors:
//
//   - HandleHTTPError: Reads error response body and returns categorized error
//   - StatusError: Error type with HTTP status code, message, and retry-after
//   - Error constructors: NewStatusError, NewAuthError, NewInternalError, etc.
//
// StatusError implements these interfaces:
//   - error: Standard error interface
//   - StatusCode() int: HTTP status code for response
//   - RetryAfter() *time.Duration: Optional retry-after hint
//   - Category() ErrorCategory: Error classification for retry logic
//
// All executors should use HandleHTTPError for consistent error handling:
//
//	if resp.StatusCode >= 400 {
//	    result := HandleHTTPError(resp, "my executor")
//	    return result.Error
//	}
package executor

import (
	"fmt"
	"io"
	"net/http"
	"time"

	cliproxyauth "github.com/nghyane/llm-mux/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
)

// HTTPErrorResult contains the result of handling an HTTP error response.
// This standardizes error handling across all executors.
type HTTPErrorResult struct {
	Error      error
	StatusCode int
	Body       []byte
}

// HandleHTTPError reads error response body and returns categorized error.
// NOTE: This function does NOT close the response body. The caller is responsible
// for closing the body (typically via defer). This avoids double-close bugs when
// callers already have defer resp.Body.Close() set up.
// Parameters:
//   - resp: HTTP response to handle
//   - executorName: Name of the executor for logging (e.g., "claude executor")
//
// Returns:
//   - HTTPErrorResult with categorized error, status code, and body
//
// All executors should use this function instead of manual error handling to ensure:
// - Consistent error categorization
// - Standardized logging
func HandleHTTPError(resp *http.Response, executorName string) HTTPErrorResult {
	body, readErr := io.ReadAll(resp.Body)

	// Handle read errors (rare but possible)
	if readErr != nil {
		return HTTPErrorResult{
			Error:      fmt.Errorf("%s: failed to read error response body: %w", executorName, readErr),
			StatusCode: resp.StatusCode,
			Body:       body,
		}
	}

	// Log the error response
	log.Debugf("%s: error status: %d, body: %s", executorName, resp.StatusCode,
		summarizeErrorBody(resp.Header.Get("Content-Type"), body))

	// Create categorized error (consistent across all executors)
	return HTTPErrorResult{
		Error:      NewStatusError(resp.StatusCode, string(body), nil),
		StatusCode: resp.StatusCode,
		Body:       body,
	}
}

// StatusError represents an error with HTTP status code, message, optional retry-after,
// and error category for consistent error handling across all executors.
type StatusError struct {
	code       int
	msg        string
	retryAfter *time.Duration
	category   cliproxyauth.ErrorCategory
}

func (e StatusError) Error() string {
	if e.msg != "" {
		return e.msg
	}
	return fmt.Sprintf("status %d", e.code)
}

func (e StatusError) StatusCode() int { return e.code }

func (e StatusError) RetryAfter() *time.Duration { return e.retryAfter }

func (e StatusError) Category() cliproxyauth.ErrorCategory { return e.category }

// Unwrap returns nil as StatusError doesn't wrap another error.
func (e StatusError) Unwrap() error { return nil }

// NewStatusError creates a StatusError with automatic category classification.
func NewStatusError(code int, msg string, retryAfter *time.Duration) StatusError {
	return StatusError{
		code:       code,
		msg:        msg,
		retryAfter: retryAfter,
		category:   cliproxyauth.CategorizeError(code, msg),
	}
}

// NewAuthError creates a StatusError for authentication failures (401).
func NewAuthError(msg string) StatusError {
	return NewStatusError(http.StatusUnauthorized, msg, nil)
}

// NewInternalError creates a StatusError for internal server errors (500).
func NewInternalError(msg string) StatusError {
	return NewStatusError(http.StatusInternalServerError, msg, nil)
}

// NewNotImplementedError creates a StatusError for not implemented features (501).
func NewNotImplementedError(msg string) StatusError {
	return NewStatusError(http.StatusNotImplemented, msg, nil)
}

// NewTimeoutError creates a StatusError for timeout errors (408).
func NewTimeoutError(msg string) StatusError {
	return NewStatusError(http.StatusRequestTimeout, msg, nil)
}
