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
	"time"

	"github.com/gin-gonic/gin"
	"github.com/nghyane/llm-mux/internal/config"
	"github.com/nghyane/llm-mux/internal/misc"
	"github.com/nghyane/llm-mux/internal/runtime/geminicli"
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
	codeAssistEndpoint = "https://cloudcode-pa.googleapis.com"
	codeAssistVersion  = "v1internal"

	// Rate limit retry settings: 5 retries with exponential backoff up to ~60 seconds total
	rateLimitMaxRetries = 5
	rateLimitBaseDelay  = 1 * time.Second  // 1s, 2s, 4s, 8s, 16s = ~31s total with exponential backoff
	rateLimitMaxDelay   = 20 * time.Second // Cap individual delay at 20s
)

var geminiOauthScopes = []string{
	"https://www.googleapis.com/auth/cloud-platform",
	"https://www.googleapis.com/auth/userinfo.email",
	"https://www.googleapis.com/auth/userinfo.profile",
}

// GeminiCLIExecutor talks to the Cloud Code Assist endpoint using OAuth credentials from auth metadata.
type GeminiCLIExecutor struct {
	cfg *config.Config
}

func NewGeminiCLIExecutor(cfg *config.Config) *GeminiCLIExecutor {
	return &GeminiCLIExecutor{cfg: cfg}
}

func (e *GeminiCLIExecutor) Identifier() string { return "gemini-cli" }

func (e *GeminiCLIExecutor) PrepareRequest(_ *http.Request, _ *cliproxyauth.Auth) error { return nil }

func (e *GeminiCLIExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	tokenSource, baseTokenData, err := prepareGeminiCLITokenSource(ctx, e.cfg, auth)
	if err != nil {
		return resp, err
	}
	reporter := newUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat

	// Translate request through canonical IR (handles all transformations internally)
	basePayload, err := TranslateToGeminiCLI(e.cfg, from, req.Model, bytes.Clone(req.Payload), false, req.Metadata)
	if err != nil {
		return resp, fmt.Errorf("failed to translate request: %w", err)
	}

	action := "generateContent"
	if req.Metadata != nil {
		if a, _ := req.Metadata["action"].(string); a == "countTokens" {
			action = "countTokens"
		}
	}

	projectID := resolveGeminiProjectID(auth)
	models := []string{req.Model}

	httpClient := newHTTPClient(ctx, e.cfg, auth, 0)

	var lastStatus int
	var lastBody []byte
	retrier := &rateLimitRetrier{}

	for idx := 0; idx < len(models); idx++ {
		attemptModel := models[idx]
		payload := append([]byte(nil), basePayload...)
		if action == "countTokens" {
			payload = deleteJSONField(payload, "project")
			payload = deleteJSONField(payload, "model")
		} else {
			payload = setJSONField(payload, "project", projectID)
			payload = setJSONField(payload, "model", attemptModel)
		}

		tok, errTok := tokenSource.Token()
		if errTok != nil {
			return resp, wrapTokenError(errTok)
		}
		updateGeminiCLITokenMetadata(auth, baseTokenData, tok)

		url := fmt.Sprintf("%s/%s:%s", codeAssistEndpoint, codeAssistVersion, action)
		if opts.Alt != "" && action != "countTokens" {
			url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
		}

		reqHTTP, errReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if errReq != nil {
			err = errReq
			return resp, err
		}
		reqHTTP.Header.Set("Content-Type", "application/json")
		reqHTTP.Header.Set("Authorization", "Bearer "+tok.AccessToken)
		applyGeminiCLIHeaders(reqHTTP)
		reqHTTP.Header.Set("Accept", "application/json")

		httpResp, errDo := httpClient.Do(reqHTTP)
		if errDo != nil {
			err = errDo
			return resp, err
		}

		data, errRead := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("gemini cli executor: close response body error: %v", errClose)
		}
		if errRead != nil {
			err = errRead
			return resp, err
		}
		if httpResp.StatusCode >= 200 && httpResp.StatusCode < 300 {
			reporter.publish(ctx, parseGeminiCLIUsage(data))

			translatedResp, err := TranslateGeminiCLIResponseNonStream(e.cfg, from, data, attemptModel)
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

		lastStatus = httpResp.StatusCode
		lastBody = append([]byte(nil), data...)
		log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		if httpResp.StatusCode == 429 {
			hasNextModel := idx+1 < len(models)
			if hasNextModel {
				log.Debugf("gemini cli executor: rate limited, retrying with next model: %s", models[idx+1])
			}
			action, ctxErr := retrier.handleRateLimit(ctx, hasNextModel, data)
			if ctxErr != nil {
				err = ctxErr
				return resp, err
			}
			switch action {
			case rateLimitActionContinue:
				continue
			case rateLimitActionRetry:
				idx--
				continue
			}
			// rateLimitActionMaxExceeded - fall through to error
		}

		err = newGeminiStatusErr(httpResp.StatusCode, data)
		return resp, err
	}

	if lastStatus == 0 {
		lastStatus = 429
	}
	err = newGeminiStatusErr(lastStatus, lastBody)
	return resp, err
}

func (e *GeminiCLIExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (stream <-chan cliproxyexecutor.StreamChunk, err error) {
	tokenSource, baseTokenData, err := prepareGeminiCLITokenSource(ctx, e.cfg, auth)
	if err != nil {
		return nil, err
	}
	reporter := newUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat

	// Translate request through canonical IR (handles all transformations internally)
	basePayload, err := TranslateToGeminiCLI(e.cfg, from, req.Model, bytes.Clone(req.Payload), true, req.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to translate request: %w", err)
	}

	projectID := resolveGeminiProjectID(auth)
	models := []string{req.Model}

	httpClient := newHTTPClient(ctx, e.cfg, auth, 0)

	var lastStatus int
	var lastBody []byte
	retrier := &rateLimitRetrier{}

	for idx, attemptModel := range models {
		payload := append([]byte(nil), basePayload...)
		payload = setJSONField(payload, "project", projectID)
		payload = setJSONField(payload, "model", attemptModel)

		tok, errTok := tokenSource.Token()
		if errTok != nil {
			return nil, wrapTokenError(errTok)
		}
		updateGeminiCLITokenMetadata(auth, baseTokenData, tok)

		url := fmt.Sprintf("%s/%s:%s", codeAssistEndpoint, codeAssistVersion, "streamGenerateContent")
		if opts.Alt == "" {
			url = url + "?alt=sse"
		} else {
			url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
		}

		reqHTTP, errReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if errReq != nil {
			err = errReq
			return nil, err
		}
		reqHTTP.Header.Set("Content-Type", "application/json")
		reqHTTP.Header.Set("Authorization", "Bearer "+tok.AccessToken)
		applyGeminiCLIHeaders(reqHTTP)
		reqHTTP.Header.Set("Accept", "text/event-stream")

		httpResp, errDo := httpClient.Do(reqHTTP)
		if errDo != nil {
			err = errDo
			return nil, err
		}
		if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
			data, errRead := io.ReadAll(httpResp.Body)
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("gemini cli executor: close response body error: %v", errClose)
			}
			if errRead != nil {
				err = errRead
				return nil, err
			}
			lastStatus = httpResp.StatusCode
			lastBody = append([]byte(nil), data...)
			log.Debugf("request error, error status: %d, error body: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
			if httpResp.StatusCode == 429 {
				hasNextModel := idx+1 < len(models)
				if hasNextModel {
					log.Debugf("gemini cli executor: rate limited, retrying with next model: %s", models[idx+1])
				}
				action, ctxErr := retrier.handleRateLimit(ctx, hasNextModel, data)
				if ctxErr != nil {
					err = ctxErr
					return nil, err
				}
				switch action {
				case rateLimitActionContinue:
					continue
				case rateLimitActionRetry:
					idx--
					continue
				}
				// rateLimitActionMaxExceeded - fall through to error
			}
			err = newGeminiStatusErr(httpResp.StatusCode, data)
			return nil, err
		}

		out := make(chan cliproxyexecutor.StreamChunk)
		stream = out
		go func(resp *http.Response, reqBody []byte, attempt string) {
			defer close(out)
			defer func() {
				if errClose := resp.Body.Close(); errClose != nil {
					log.Errorf("gemini cli executor: close response body error: %v", errClose)
				}
			}()
			if opts.Alt == "" {
				scanner := bufio.NewScanner(resp.Body)
				scanner.Buffer(make([]byte, 64*1024), DefaultStreamBufferSize)
				streamState := &GeminiCLIStreamState{
					ClaudeState: from_ir.NewClaudeStreamState(),
				}

				for scanner.Scan() {
					line := scanner.Bytes()
					if detail, ok := parseGeminiCLIStreamUsage(line); ok {
						reporter.publish(ctx, detail)
					}
					if bytes.HasPrefix(line, dataTag) {
						messageID := "chatcmpl-" + attempt
						translatedChunks, err := TranslateGeminiCLIResponseStream(e.cfg, from, bytes.Clone(line), attempt, messageID, streamState)
						if err != nil {
							out <- cliproxyexecutor.StreamChunk{Err: err}
							return
						}
						for _, chunk := range translatedChunks {
							out <- cliproxyexecutor.StreamChunk{Payload: chunk}
						}
					}
				}
				if errScan := scanner.Err(); errScan != nil {
					reporter.publishFailure(ctx)
					out <- cliproxyexecutor.StreamChunk{Err: errScan}
				}
				return
			}

			data, errRead := io.ReadAll(resp.Body)
			if errRead != nil {
				reporter.publishFailure(ctx)
				out <- cliproxyexecutor.StreamChunk{Err: errRead}
				return
			}
			reporter.publish(ctx, parseGeminiCLIUsage(data))

			// For non-streaming responses, convert to non-stream and return as single chunk
			translatedResp, err := TranslateGeminiCLIResponseNonStream(e.cfg, from, data, attempt)
			if err != nil {
				out <- cliproxyexecutor.StreamChunk{Err: err}
				return
			}
			if translatedResp != nil {
				out <- cliproxyexecutor.StreamChunk{Payload: translatedResp}
			} else {
				out <- cliproxyexecutor.StreamChunk{Payload: data}
			}
		}(httpResp, append([]byte(nil), payload...), attemptModel)

		return stream, nil
	}

	if lastStatus == 0 {
		lastStatus = 429
	}
	err = newGeminiStatusErr(lastStatus, lastBody)
	return nil, err
}

func (e *GeminiCLIExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	tokenSource, baseTokenData, err := prepareGeminiCLITokenSource(ctx, e.cfg, auth)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	from := opts.SourceFormat
	models := []string{req.Model}

	httpClient := newHTTPClient(ctx, e.cfg, auth, 0)
	respCtx := context.WithValue(ctx, "alt", opts.Alt)

	var lastStatus int
	var lastBody []byte
	retrier := &rateLimitRetrier{}

	for idx, attemptModel := range models {
		// Translate request through canonical IR
		payload, errTranslate := TranslateToGeminiCLI(e.cfg, from, attemptModel, bytes.Clone(req.Payload), false, req.Metadata)
		if errTranslate != nil {
			return cliproxyexecutor.Response{}, fmt.Errorf("failed to translate request: %w", errTranslate)
		}

		// Remove fields not needed for countTokens
		payload = deleteJSONField(payload, "project")
		payload = deleteJSONField(payload, "model")
		payload = deleteJSONField(payload, "request.safetySettings")

		tok, errTok := tokenSource.Token()
		if errTok != nil {
			return cliproxyexecutor.Response{}, wrapTokenError(errTok)
		}
		updateGeminiCLITokenMetadata(auth, baseTokenData, tok)

		url := fmt.Sprintf("%s/%s:%s", codeAssistEndpoint, codeAssistVersion, "countTokens")
		if opts.Alt != "" {
			url = url + fmt.Sprintf("?$alt=%s", opts.Alt)
		}

		reqHTTP, errReq := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if errReq != nil {
			return cliproxyexecutor.Response{}, errReq
		}
		reqHTTP.Header.Set("Content-Type", "application/json")
		reqHTTP.Header.Set("Authorization", "Bearer "+tok.AccessToken)
		applyGeminiCLIHeaders(reqHTTP)
		reqHTTP.Header.Set("Accept", "application/json")

		resp, errDo := httpClient.Do(reqHTTP)
		if errDo != nil {
			return cliproxyexecutor.Response{}, errDo
		}
		data, errRead := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if errRead != nil {
			return cliproxyexecutor.Response{}, errRead
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			count := gjson.GetBytes(data, "totalTokens").Int()
			formatGeminiCLI := sdktranslator.FromString("gemini-cli")
			translated := sdktranslator.TranslateTokenCount(respCtx, formatGeminiCLI, from, count, data)
			return cliproxyexecutor.Response{Payload: []byte(translated)}, nil
		}
		lastStatus = resp.StatusCode
		lastBody = append([]byte(nil), data...)
		if resp.StatusCode == 429 {
			hasNextModel := idx+1 < len(models)
			if hasNextModel {
				log.Debugf("gemini cli executor: rate limited, retrying with next model")
			}
			action, ctxErr := retrier.handleRateLimit(ctx, hasNextModel, data)
			if ctxErr != nil {
				return cliproxyexecutor.Response{}, ctxErr
			}
			switch action {
			case rateLimitActionContinue:
				continue
			case rateLimitActionRetry:
				idx--
				continue
			}
			// rateLimitActionMaxExceeded - fall through to break
		}
		break
	}

	if lastStatus == 0 {
		lastStatus = 429
	}
	return cliproxyexecutor.Response{}, newGeminiStatusErr(lastStatus, lastBody)
}

func (e *GeminiCLIExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("gemini cli executor: refresh called")
	_ = ctx
	return auth, nil
}

func prepareGeminiCLITokenSource(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth) (oauth2.TokenSource, map[string]any, error) {
	metadata := geminiOAuthMetadata(auth)
	if auth == nil || metadata == nil {
		return nil, nil, fmt.Errorf("gemini-cli auth metadata missing")
	}

	var base map[string]any
	if tokenRaw, ok := metadata["token"].(map[string]any); ok && tokenRaw != nil {
		base = cloneMap(tokenRaw)
	} else {
		base = make(map[string]any)
	}

	var token oauth2.Token
	if len(base) > 0 {
		if raw, err := json.Marshal(base); err == nil {
			_ = json.Unmarshal(raw, &token)
		}
	}

	if token.AccessToken == "" {
		token.AccessToken = stringValue(metadata, "access_token")
	}
	if token.RefreshToken == "" {
		token.RefreshToken = stringValue(metadata, "refresh_token")
	}
	if token.TokenType == "" {
		token.TokenType = stringValue(metadata, "token_type")
	}
	if token.Expiry.IsZero() {
		if expiry := stringValue(metadata, "expiry"); expiry != "" {
			if ts, err := time.Parse(time.RFC3339, expiry); err == nil {
				token.Expiry = ts
			}
		}
	}

	conf := &oauth2.Config{
		ClientID:     geminiOauthClientID,
		ClientSecret: geminiOauthClientSecret,
		Scopes:       geminiOauthScopes,
		Endpoint:     google.Endpoint,
	}

	ctxToken := ctx
	if httpClient := newProxyAwareHTTPClient(ctx, cfg, auth, 0); httpClient != nil {
		ctxToken = context.WithValue(ctxToken, oauth2.HTTPClient, httpClient)
	}

	src := conf.TokenSource(ctxToken, &token)
	currentToken, err := src.Token()
	if err != nil {
		return nil, nil, wrapTokenError(err)
	}
	updateGeminiCLITokenMetadata(auth, base, currentToken)
	return oauth2.ReuseTokenSource(currentToken, src), base, nil
}

func updateGeminiCLITokenMetadata(auth *cliproxyauth.Auth, base map[string]any, tok *oauth2.Token) {
	if auth == nil || tok == nil {
		return
	}
	merged := buildGeminiTokenMap(base, tok)
	fields := buildGeminiTokenFields(tok, merged)
	shared := geminicli.ResolveSharedCredential(auth.Runtime)
	if shared != nil {
		snapshot := shared.MergeMetadata(fields)
		if !geminicli.IsVirtual(auth.Runtime) {
			auth.Metadata = snapshot
		}
		return
	}
	if auth.Metadata == nil {
		auth.Metadata = make(map[string]any)
	}
	for k, v := range fields {
		auth.Metadata[k] = v
	}
}

func buildGeminiTokenMap(base map[string]any, tok *oauth2.Token) map[string]any {
	merged := cloneMap(base)
	if merged == nil {
		merged = make(map[string]any)
	}
	if raw, err := json.Marshal(tok); err == nil {
		var tokenMap map[string]any
		if err = json.Unmarshal(raw, &tokenMap); err == nil {
			for k, v := range tokenMap {
				merged[k] = v
			}
		}
	}
	return merged
}

func buildGeminiTokenFields(tok *oauth2.Token, merged map[string]any) map[string]any {
	fields := make(map[string]any, 5)
	if tok.AccessToken != "" {
		fields["access_token"] = tok.AccessToken
	}
	if tok.TokenType != "" {
		fields["token_type"] = tok.TokenType
	}
	if tok.RefreshToken != "" {
		fields["refresh_token"] = tok.RefreshToken
	}
	if !tok.Expiry.IsZero() {
		fields["expiry"] = tok.Expiry.Format(time.RFC3339)
	}
	if len(merged) > 0 {
		fields["token"] = cloneMap(merged)
	}
	return fields
}

func resolveGeminiProjectID(auth *cliproxyauth.Auth) string {
	if auth == nil {
		return ""
	}
	if runtime := auth.Runtime; runtime != nil {
		if virtual, ok := runtime.(*geminicli.VirtualCredential); ok && virtual != nil {
			return strings.TrimSpace(virtual.ProjectID)
		}
	}
	return strings.TrimSpace(stringValue(auth.Metadata, "project_id"))
}

func geminiOAuthMetadata(auth *cliproxyauth.Auth) map[string]any {
	if auth == nil {
		return nil
	}
	if shared := geminicli.ResolveSharedCredential(auth.Runtime); shared != nil {
		if snapshot := shared.MetadataSnapshot(); len(snapshot) > 0 {
			return snapshot
		}
	}
	return auth.Metadata
}

func newHTTPClient(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth, timeout time.Duration) *http.Client {
	return newProxyAwareHTTPClient(ctx, cfg, auth, timeout)
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func stringValue(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key]; ok {
		switch typed := v.(type) {
		case string:
			return typed
		case fmt.Stringer:
			return typed.String()
		}
	}
	return ""
}

// applyGeminiCLIHeaders sets required headers for the Gemini CLI upstream.
func applyGeminiCLIHeaders(r *http.Request) {
	var ginHeaders http.Header
	if ginCtx, ok := r.Context().Value("gin").(*gin.Context); ok && ginCtx != nil && ginCtx.Request != nil {
		ginHeaders = ginCtx.Request.Header
	}

	misc.EnsureHeader(r.Header, ginHeaders, "User-Agent", "google-api-nodejs-client/9.15.1")
	misc.EnsureHeader(r.Header, ginHeaders, "X-Goog-Api-Client", "gl-node/22.17.0")
	misc.EnsureHeader(r.Header, ginHeaders, "Client-Metadata", geminiCLIClientMetadata())
}

// geminiCLIClientMetadata returns a compact metadata string required by upstream.
func geminiCLIClientMetadata() string {
	// Keep parity with CLI client defaults
	return "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI"
}

// setJSONField sets a top-level JSON field on a byte slice payload via sjson.
func setJSONField(body []byte, key, value string) []byte {
	if key == "" {
		return body
	}
	updated, err := sjson.SetBytes(body, key, value)
	if err != nil {
		return body
	}
	return updated
}

// deleteJSONField removes a top-level key if present (best-effort) via sjson.
func deleteJSONField(body []byte, key string) []byte {
	if key == "" || len(body) == 0 {
		return body
	}
	updated, err := sjson.DeleteBytes(body, key)
	if err != nil {
		return body
	}
	return updated
}

func fixGeminiCLIImageAspectRatio(modelName string, rawJSON []byte) []byte {
	if modelName == "gemini-2.5-flash-image-preview" {
		aspectRatioResult := gjson.GetBytes(rawJSON, "request.generationConfig.imageConfig.aspectRatio")
		if aspectRatioResult.Exists() {
			contents := gjson.GetBytes(rawJSON, "request.contents")
			contentArray := contents.Array()
			if len(contentArray) > 0 {
				hasInlineData := false
			loopContent:
				for _, content := range contentArray {
					parts := content.Get("parts").Array()
					for _, part := range parts {
						if part.Get("inlineData").Exists() {
							hasInlineData = true
							break loopContent
						}
					}
				}

				if !hasInlineData {
					emptyImageBase64ed, _ := util.CreateWhiteImageBase64(aspectRatioResult.String())
					emptyImagePart := `{"inlineData":{"mime_type":"image/png","data":""}}`
					emptyImagePart, _ = sjson.Set(emptyImagePart, "inlineData.data", emptyImageBase64ed)
					newPartsJson := `[]`
					newPartsJson, _ = sjson.SetRaw(newPartsJson, "-1", `{"text": "Based on the following requirements, create an image within the uploaded picture. The new content *MUST* completely cover the entire area of the original picture, maintaining its exact proportions, and *NO* blank areas should appear."}`)
					newPartsJson, _ = sjson.SetRaw(newPartsJson, "-1", emptyImagePart)

					parts := contentArray[0].Get("parts").Array()
					for _, part := range parts {
						newPartsJson, _ = sjson.SetRaw(newPartsJson, "-1", part.Raw)
					}

					rawJSON, _ = sjson.SetRawBytes(rawJSON, "request.contents.0.parts", []byte(newPartsJson))
					rawJSON, _ = sjson.SetRawBytes(rawJSON, "request.generationConfig.responseModalities", []byte(`["IMAGE", "TEXT"]`))
				}
			}
			rawJSON, _ = sjson.DeleteBytes(rawJSON, "request.generationConfig.imageConfig")
		}
	}
	return rawJSON
}

func newGeminiStatusErr(statusCode int, body []byte) statusErr {
	err := statusErr{code: statusCode, msg: string(body)}
	if statusCode == http.StatusTooManyRequests {
		if retryAfter, parseErr := parseRetryDelay(body); parseErr == nil && retryAfter != nil {
			err.retryAfter = retryAfter
		}
	}
	return err
}

// wrapTokenError wraps token retrieval errors with proper categorization
func wrapTokenError(err error) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	return newCategorizedError(http.StatusUnauthorized, msg, nil)
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
		log.Debug("gemini cli executor: rate limited, max retries exceeded")
		return rateLimitActionMaxExceeded, nil
	}

	delay := r.calculateDelay(errorBody)
	r.retryCount++
	log.Debugf("gemini cli executor: rate limited, waiting %v before retry %d/%d", delay, r.retryCount, rateLimitMaxRetries)

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
		if delay > rateLimitMaxDelay {
			delay = rateLimitMaxDelay
		}
		return delay
	}

	// Fall back to exponential backoff: baseDelay * 2^retryCount
	delay := rateLimitBaseDelay * time.Duration(1<<r.retryCount)
	if delay > rateLimitMaxDelay {
		delay = rateLimitMaxDelay
	}
	return delay
}

// parseRetryDelay extracts the retry delay from a Google API 429 error response.
// The error response contains a RetryInfo.retryDelay field in the format "0.847655010s".
// Returns the parsed duration or an error if it cannot be determined.
func parseRetryDelay(errorBody []byte) (*time.Duration, error) {
	// Try to parse the retryDelay from the error response
	// Format: error.details[].retryDelay where @type == "type.googleapis.com/google.rpc.RetryInfo"
	details := gjson.GetBytes(errorBody, "error.details")
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
					return nil, fmt.Errorf("failed to parse duration")
				}
				return &duration, nil
			}
		}
	}

	return nil, fmt.Errorf("no RetryInfo found")
}
