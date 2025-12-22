// Package executor contains provider executors. This file implements the Vertex AI
// Gemini executor that talks to Google Vertex AI endpoints using service account
// credentials imported by the CLI.
package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	vertexauth "github.com/nghyane/llm-mux/internal/auth/vertex"
	"github.com/nghyane/llm-mux/internal/config"
	"github.com/nghyane/llm-mux/internal/translator/from_ir"
	"github.com/nghyane/llm-mux/internal/util"
	cliproxyauth "github.com/nghyane/llm-mux/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/nghyane/llm-mux/sdk/cliproxy/executor"
	sdktranslator "github.com/nghyane/llm-mux/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	// vertexAPIVersion aligns with current public Vertex Generative AI API.
	vertexAPIVersion = "v1"
)

// GeminiVertexExecutor sends requests to Vertex AI Gemini endpoints using service account credentials.
type GeminiVertexExecutor struct {
	cfg *config.Config
}

// NewGeminiVertexExecutor constructs the Vertex executor.
func NewGeminiVertexExecutor(cfg *config.Config) *GeminiVertexExecutor {
	return &GeminiVertexExecutor{cfg: cfg}
}

// Identifier returns provider key for manager routing.
func (e *GeminiVertexExecutor) Identifier() string { return "vertex" }

// PrepareRequest is a no-op for Vertex.
func (e *GeminiVertexExecutor) PrepareRequest(_ *http.Request, _ *cliproxyauth.Auth) error {
	return nil
}

// Execute handles non-streaming requests.
func (e *GeminiVertexExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	// Try API key authentication first
	apiKey, baseURL := vertexAPICreds(auth)

	// If no API key found, fall back to service account authentication
	if apiKey == "" {
		projectID, location, saJSON, errCreds := vertexCreds(auth)
		if errCreds != nil {
			return resp, errCreds
		}
		return e.executeWithServiceAccount(ctx, auth, req, opts, projectID, location, saJSON)
	}

	// Use API key authentication
	return e.executeWithAPIKey(ctx, auth, req, opts, apiKey, baseURL)
}

// ExecuteStream handles SSE streaming for Vertex.
func (e *GeminiVertexExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	// Try API key authentication first
	apiKey, baseURL := vertexAPICreds(auth)

	// If no API key found, fall back to service account authentication
	if apiKey == "" {
		projectID, location, saJSON, errCreds := vertexCreds(auth)
		if errCreds != nil {
			return nil, errCreds
		}
		return e.executeStreamWithServiceAccount(ctx, auth, req, opts, projectID, location, saJSON)
	}

	// Use API key authentication
	return e.executeStreamWithAPIKey(ctx, auth, req, opts, apiKey, baseURL)
}

// CountTokens calls Vertex countTokens endpoint.
func (e *GeminiVertexExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	// Try API key authentication first
	apiKey, baseURL := vertexAPICreds(auth)

	// If no API key found, fall back to service account authentication
	if apiKey == "" {
		projectID, location, saJSON, errCreds := vertexCreds(auth)
		if errCreds != nil {
			return cliproxyexecutor.Response{}, errCreds
		}
		return e.countTokensWithServiceAccount(ctx, auth, req, opts, projectID, location, saJSON)
	}

	// Use API key authentication
	return e.countTokensWithAPIKey(ctx, auth, req, opts, apiKey, baseURL)
}

// countTokensWithServiceAccount handles token counting using service account credentials.
func (e *GeminiVertexExecutor) countTokensWithServiceAccount(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, projectID, location string, saJSON []byte) (cliproxyexecutor.Response, error) {
	from := opts.SourceFormat
	// Thinking config is handled by translator
	translatedReq, err := TranslateToGemini(e.cfg, from, req.Model, bytes.Clone(req.Payload), false, req.Metadata)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	translatedReq = util.StripThinkingConfigIfUnsupported(req.Model, translatedReq)
	respCtx := context.WithValue(ctx, altContextKey{}, opts.Alt)
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "tools")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "generationConfig")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "safetySettings")

	baseURL := vertexBaseURL(location)
	url := fmt.Sprintf("%s/%s/projects/%s/locations/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, projectID, location, req.Model, "countTokens")

	httpReq, errNewReq := http.NewRequestWithContext(respCtx, http.MethodPost, url, bytes.NewReader(translatedReq))
	if errNewReq != nil {
		return cliproxyexecutor.Response{}, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if token, errTok := vertexAccessToken(ctx, e.cfg, auth, saJSON); errTok == nil && token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+token)
	} else if errTok != nil {
		log.Errorf("vertex executor: access token error: %v", errTok)
		return cliproxyexecutor.Response{}, statusErr{code: 500, msg: "internal server error"}
	}
	applyGeminiHeaders(httpReq, auth)

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		return cliproxyexecutor.Response{}, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		result := HandleHTTPError(httpResp, "gemini-vertex executor")
		return cliproxyexecutor.Response{}, result.Error
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		return cliproxyexecutor.Response{}, errRead
	}
	count := gjson.GetBytes(data, "totalTokens").Int()
	to := formatGemini
	out := sdktranslator.TranslateTokenCount(ctx, to, from, count, data)
	return cliproxyexecutor.Response{Payload: []byte(out)}, nil
}

// countTokensWithAPIKey handles token counting using API key credentials.
func (e *GeminiVertexExecutor) countTokensWithAPIKey(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, apiKey, baseURL string) (cliproxyexecutor.Response, error) {
	from := opts.SourceFormat
	// Thinking config is handled by translator
	translatedReq, err := TranslateToGemini(e.cfg, from, req.Model, bytes.Clone(req.Payload), false, req.Metadata)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	translatedReq = util.StripThinkingConfigIfUnsupported(req.Model, translatedReq)
	respCtx := context.WithValue(ctx, altContextKey{}, opts.Alt)
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "tools")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "generationConfig")
	translatedReq, _ = sjson.DeleteBytes(translatedReq, "safetySettings")

	// For API key auth, use simpler URL format without project/location
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, req.Model, "countTokens")

	httpReq, errNewReq := http.NewRequestWithContext(respCtx, http.MethodPost, url, bytes.NewReader(translatedReq))
	if errNewReq != nil {
		return cliproxyexecutor.Response{}, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", apiKey)
	}
	applyGeminiHeaders(httpReq, auth)

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		return cliproxyexecutor.Response{}, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		result := HandleHTTPError(httpResp, "gemini-vertex executor")
		return cliproxyexecutor.Response{}, result.Error
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		return cliproxyexecutor.Response{}, errRead
	}
	count := gjson.GetBytes(data, "totalTokens").Int()
	to := formatGemini
	out := sdktranslator.TranslateTokenCount(ctx, to, from, count, data)
	return cliproxyexecutor.Response{Payload: []byte(out)}, nil
}

// Refresh is a no-op for service account based credentials.
func (e *GeminiVertexExecutor) Refresh(_ context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	return auth, nil
}

// executeWithServiceAccount handles authentication using service account credentials.
// This method contains the original service account authentication logic.
func (e *GeminiVertexExecutor) executeWithServiceAccount(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, projectID, location string, saJSON []byte) (resp cliproxyexecutor.Response, err error) {
	reporter := newUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	// Thinking config is now handled by the translator (including auto-apply for Claude)
	body, err := TranslateToGemini(e.cfg, from, req.Model, bytes.Clone(req.Payload), false, req.Metadata)
	if err != nil {
		return resp, err
	}
	body = util.StripThinkingConfigIfUnsupported(req.Model, body)

	action := "generateContent"
	if req.Metadata != nil {
		if a, _ := req.Metadata["action"].(string); a == "countTokens" {
			action = "countTokens"
		}
	}

	url := fmt.Sprintf("https://%s-aiplatform.googleapis.com/%s/projects/%s/locations/%s/publishers/google/models/%s:%s", location, vertexAPIVersion, projectID, location, req.Model, action)
	if opts.Alt != "" && action != "countTokens" {
		url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
	}

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return resp, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if token, errTok := vertexAccessToken(ctx, e.cfg, auth, saJSON); errTok == nil && token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+token)
	} else if errTok != nil {
		log.Errorf("vertex executor: access token error: %v", errTok)
		return resp, statusErr{code: 500, msg: "internal server error"}
	}
	applyGeminiHeaders(httpReq, auth)

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		result := HandleHTTPError(httpResp, "gemini-vertex executor")
		return resp, result.Error
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		return resp, errRead
	}
	reporter.publish(ctx, extractUsageFromGeminiResponse(data))

	translatedResp, err := TranslateGeminiResponseNonStream(e.cfg, from, data, req.Model)
	if err != nil {
		return resp, err
	}
	if translatedResp != nil {
		resp = cliproxyexecutor.Response{Payload: translatedResp}
	} else {
		resp = cliproxyexecutor.Response{Payload: data}
	}
	return resp, nil
}

// executeWithAPIKey handles authentication using API key credentials.
func (e *GeminiVertexExecutor) executeWithAPIKey(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, apiKey, baseURL string) (resp cliproxyexecutor.Response, err error) {
	reporter := newUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	// Thinking config is now handled by the translator (including auto-apply for Claude)
	body, err := TranslateToGemini(e.cfg, from, req.Model, bytes.Clone(req.Payload), false, req.Metadata)
	if err != nil {
		return resp, err
	}
	body = util.StripThinkingConfigIfUnsupported(req.Model, body)

	action := "generateContent"
	if req.Metadata != nil {
		if a, _ := req.Metadata["action"].(string); a == "countTokens" {
			action = "countTokens"
		}
	}

	// For API key auth, use simpler URL format without project/location
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, req.Model, action)
	if opts.Alt != "" && action != "countTokens" {
		url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return resp, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", apiKey)
	}
	applyGeminiHeaders(httpReq, auth)

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vertex executor: close response body error: %v", errClose)
		}
	}()
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		result := HandleHTTPError(httpResp, "gemini-vertex executor")
		return resp, result.Error
	}
	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		return resp, errRead
	}
	reporter.publish(ctx, extractUsageFromGeminiResponse(data))

	translatedResp, err := TranslateGeminiResponseNonStream(e.cfg, from, data, req.Model)
	if err != nil {
		return resp, err
	}
	if translatedResp != nil {
		resp = cliproxyexecutor.Response{Payload: translatedResp}
	} else {
		resp = cliproxyexecutor.Response{Payload: data}
	}
	return resp, nil
}

// executeStreamWithServiceAccount handles streaming authentication using service account credentials.
func (e *GeminiVertexExecutor) executeStreamWithServiceAccount(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, projectID, location string, saJSON []byte) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	reporter := newUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	// Thinking config is now handled by the translator (including auto-apply for Claude)
	translation, err := TranslateToGeminiWithTokens(e.cfg, from, req.Model, bytes.Clone(req.Payload), true, req.Metadata)
	if err != nil {
		return nil, err
	}
	body := translation.Payload
	body = util.StripThinkingConfigIfUnsupported(req.Model, body)

	baseURL := vertexBaseURL(location)
	url := fmt.Sprintf("%s/%s/projects/%s/locations/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, projectID, location, req.Model, "streamGenerateContent")
	if opts.Alt == "" {
		url = url + "?alt=sse"
	} else {
		url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return nil, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if token, errTok := vertexAccessToken(ctx, e.cfg, auth, saJSON); errTok == nil && token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+token)
	} else if errTok != nil {
		log.Errorf("vertex executor: access token error: %v", errTok)
		return nil, statusErr{code: 500, msg: "internal server error"}
	}
	applyGeminiHeaders(httpReq, auth)

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		return nil, errDo
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		result := HandleHTTPError(httpResp, "gemini-vertex executor")
		return nil, result.Error
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	stream = out
	messageID := "chatcmpl-" + req.Model
	estimatedInputTokens := translation.EstimatedInputTokens

	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("vertex executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 64*1024), DefaultStreamBufferSize)
		streamState := &GeminiCLIStreamState{
			ClaudeState: from_ir.NewClaudeStreamState(),
		}
		// Set pre-calculated input tokens for message_start
		streamState.ClaudeState.EstimatedInputTokens = estimatedInputTokens

		for scanner.Scan() {
			line := scanner.Bytes()
			result, err := TranslateGeminiResponseStreamWithUsage(e.cfg, from, bytes.Clone(line), req.Model, messageID, streamState)
			if err != nil {
				out <- cliproxyexecutor.StreamChunk{Err: err}
				return
			}
			if result.Usage != nil {
				reporter.publish(ctx, result.Usage)
			}
			for _, chunk := range result.Chunks {
				out <- cliproxyexecutor.StreamChunk{Payload: chunk}
			}
		}
		// Handle [DONE]
		doneResult, err := TranslateGeminiResponseStreamWithUsage(e.cfg, from, []byte("[DONE]"), req.Model, messageID, streamState)
		if err != nil {
			out <- cliproxyexecutor.StreamChunk{Err: err}
			return
		}
		for _, chunk := range doneResult.Chunks {
			out <- cliproxyexecutor.StreamChunk{Payload: chunk}
		}
		if errScan := scanner.Err(); errScan != nil {
			reporter.publishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		}
	}()
	return stream, nil
}

// executeStreamWithAPIKey handles streaming authentication using API key credentials.
func (e *GeminiVertexExecutor) executeStreamWithAPIKey(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, apiKey, baseURL string) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	reporter := newUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	// Thinking config is now handled by the translator (including auto-apply for Claude)
	translation, err := TranslateToGeminiWithTokens(e.cfg, from, req.Model, bytes.Clone(req.Payload), true, req.Metadata)
	if err != nil {
		return nil, err
	}
	body := translation.Payload
	body = util.StripThinkingConfigIfUnsupported(req.Model, body)

	// For API key auth, use simpler URL format without project/location
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	url := fmt.Sprintf("%s/%s/publishers/google/models/%s:%s", baseURL, vertexAPIVersion, req.Model, "streamGenerateContent")
	if opts.Alt == "" {
		url = url + "?alt=sse"
	} else {
		url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
	}
	body, _ = sjson.DeleteBytes(body, "session_id")

	httpReq, errNewReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if errNewReq != nil {
		return nil, errNewReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-goog-api-key", apiKey)
	}
	applyGeminiHeaders(httpReq, auth)

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		return nil, errDo
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		result := HandleHTTPError(httpResp, "gemini-vertex executor")
		return nil, result.Error
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	stream = out
	messageID := "chatcmpl-" + req.Model
	estimatedInputTokens := translation.EstimatedInputTokens

	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("vertex executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 64*1024), DefaultStreamBufferSize)
		streamState := &GeminiCLIStreamState{
			ClaudeState: from_ir.NewClaudeStreamState(),
		}
		// Set pre-calculated input tokens for message_start
		streamState.ClaudeState.EstimatedInputTokens = estimatedInputTokens

		for scanner.Scan() {
			line := scanner.Bytes()
			result, err := TranslateGeminiResponseStreamWithUsage(e.cfg, from, bytes.Clone(line), req.Model, messageID, streamState)
			if err != nil {
				out <- cliproxyexecutor.StreamChunk{Err: err}
				return
			}
			if result.Usage != nil {
				reporter.publish(ctx, result.Usage)
			}
			for _, chunk := range result.Chunks {
				out <- cliproxyexecutor.StreamChunk{Payload: chunk}
			}
		}
		// Handle [DONE]
		doneResult, err := TranslateGeminiResponseStreamWithUsage(e.cfg, from, []byte("[DONE]"), req.Model, messageID, streamState)
		if err != nil {
			out <- cliproxyexecutor.StreamChunk{Err: err}
			return
		}
		for _, chunk := range doneResult.Chunks {
			out <- cliproxyexecutor.StreamChunk{Payload: chunk}
		}
		if errScan := scanner.Err(); errScan != nil {
			reporter.publishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		}
	}()
	return stream, nil
}

// vertexCreds extracts project, location and raw service account JSON from auth metadata.
func vertexCreds(a *cliproxyauth.Auth) (projectID, location string, serviceAccountJSON []byte, err error) {
	if a == nil || a.Metadata == nil {
		return "", "", nil, fmt.Errorf("vertex executor: missing auth metadata")
	}
	if v, ok := a.Metadata["project_id"].(string); ok {
		projectID = strings.TrimSpace(v)
	}
	if projectID == "" {
		// Some service accounts may use "project"; still prefer standard field
		if v, ok := a.Metadata["project"].(string); ok {
			projectID = strings.TrimSpace(v)
		}
	}
	if projectID == "" {
		return "", "", nil, fmt.Errorf("vertex executor: missing project_id in credentials")
	}
	if v, ok := a.Metadata["location"].(string); ok && strings.TrimSpace(v) != "" {
		location = strings.TrimSpace(v)
	} else {
		location = "us-central1"
	}
	var sa map[string]any
	if raw, ok := a.Metadata["service_account"].(map[string]any); ok {
		sa = raw
	}
	if sa == nil {
		return "", "", nil, fmt.Errorf("vertex executor: missing service_account in credentials")
	}
	normalized, errNorm := vertexauth.NormalizeServiceAccountMap(sa)
	if errNorm != nil {
		return "", "", nil, fmt.Errorf("vertex executor: %w", errNorm)
	}
	saJSON, errMarshal := json.Marshal(normalized)
	if errMarshal != nil {
		return "", "", nil, fmt.Errorf("vertex executor: marshal service_account failed: %w", errMarshal)
	}
	return projectID, location, saJSON, nil
}

// vertexAPICreds extracts API key and base URL from auth attributes following the claudeCreds pattern.
func vertexAPICreds(a *cliproxyauth.Auth) (apiKey, baseURL string) {
	if a == nil {
		return "", ""
	}
	if a.Attributes != nil {
		apiKey = a.Attributes["api_key"]
		baseURL = a.Attributes["base_url"]
	}
	if apiKey == "" && a.Metadata != nil {
		if v, ok := a.Metadata["access_token"].(string); ok {
			apiKey = v
		}
	}
	return
}

func vertexBaseURL(location string) string {
	loc := strings.TrimSpace(location)
	if loc == "" {
		loc = "us-central1"
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com", loc)
}

func vertexAccessToken(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth, saJSON []byte) (string, error) {
	if httpClient := newProxyAwareHTTPClient(ctx, cfg, auth, 0); httpClient != nil {
		ctx = context.WithValue(ctx, oauth2.HTTPClient, httpClient)
	}
	// Use cloud-platform scope for Vertex AI.
	creds, errCreds := google.CredentialsFromJSON(ctx, saJSON, "https://www.googleapis.com/auth/cloud-platform")
	if errCreds != nil {
		return "", fmt.Errorf("vertex executor: parse service account json failed: %w", errCreds)
	}
	tok, errTok := creds.TokenSource.Token()
	if errTok != nil {
		return "", fmt.Errorf("vertex executor: get access token failed: %w", errTok)
	}
	return tok.AccessToken, nil
}
