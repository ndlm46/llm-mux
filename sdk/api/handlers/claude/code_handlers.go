package claude

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/nghyane/llm-mux/internal/constant"
	"github.com/nghyane/llm-mux/internal/interfaces"
	"github.com/nghyane/llm-mux/internal/registry"
	"github.com/nghyane/llm-mux/sdk/api/handlers"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

type ClaudeCodeAPIHandler struct {
	*handlers.BaseAPIHandler
}

func NewClaudeCodeAPIHandler(apiHandlers *handlers.BaseAPIHandler) *ClaudeCodeAPIHandler {
	return &ClaudeCodeAPIHandler{
		BaseAPIHandler: apiHandlers,
	}
}

func (h *ClaudeCodeAPIHandler) HandlerType() string {
	return constant.Claude
}

func (h *ClaudeCodeAPIHandler) Models() []map[string]any {
	return registry.GetGlobalRegistry().GetAvailableModels("claude")
}

func (h *ClaudeCodeAPIHandler) ClaudeMessages(c *gin.Context) {
	rawJSON, err := c.GetRawData()
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	streamResult := gjson.GetBytes(rawJSON, "stream")
	if !streamResult.Exists() || streamResult.Type == gjson.False {
		h.handleNonStreamingResponse(c, rawJSON)
	} else {
		h.handleStreamingResponse(c, rawJSON)
	}
}

func (h *ClaudeCodeAPIHandler) ClaudeCountTokens(c *gin.Context) {
	rawJSON, err := c.GetRawData()
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	c.Header("Content-Type", "application/json")

	alt := h.GetAlt(c)
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())

	modelName := gjson.GetBytes(rawJSON, "model").String()

	resp, errMsg := h.ExecuteCountWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, alt)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

func (h *ClaudeCodeAPIHandler) ClaudeModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"data": h.Models(),
	})
}

func (h *ClaudeCodeAPIHandler) handleNonStreamingResponse(c *gin.Context, rawJSON []byte) {
	c.Header("Content-Type", "application/json")
	alt := h.GetAlt(c)
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())

	modelName := gjson.GetBytes(rawJSON, "model").String()

	resp, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, alt)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}

	// Decompress gzipped responses - Claude API sometimes returns gzip without Content-Encoding header
	// This fixes title generation and other non-streaming responses that arrive compressed
	if len(resp) >= 2 && resp[0] == 0x1f && resp[1] == 0x8b {
		gzReader, err := gzip.NewReader(bytes.NewReader(resp))
		if err != nil {
			log.Warnf("failed to decompress gzipped Claude response: %v", err)
		} else {
			defer gzReader.Close()
			if decompressed, err := io.ReadAll(gzReader); err != nil {
				log.Warnf("failed to read decompressed Claude response: %v", err)
			} else {
				resp = decompressed
			}
		}
	}

	_, _ = c.Writer.Write(resp)
	cliCancel()
}

func (h *ClaudeCodeAPIHandler) handleStreamingResponse(c *gin.Context, rawJSON []byte) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported",
				Type:    "server_error",
			},
		})
		return
	}

	modelName := gjson.GetBytes(rawJSON, "model").String()
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	dataChan, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "")
	h.forwardClaudeStream(c, flusher, func(err error) { cliCancel(err) }, dataChan, errChan)
}

func (h *ClaudeCodeAPIHandler) forwardClaudeStream(c *gin.Context, flusher http.Flusher, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
	writer := bufio.NewWriterSize(c.Writer, 16*1024)
	ticker := time.NewTicker(120 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-c.Request.Context().Done():
			_ = writer.Flush()
			cancel(c.Request.Context().Err())
			return

		case <-ticker.C:
			if writer.Buffered() >= 8*1024 {
				if err := writer.Flush(); err != nil {
					cancel(err)
					return
				}
				flusher.Flush()
			}

		case chunk, ok := <-data:
			if !ok {
				_ = writer.Flush()
				cancel(nil)
				return
			}
			if len(chunk) > 0 {
				_, _ = writer.Write(chunk)
			}

		case errMsg, ok := <-errs:
			if !ok {
				continue
			}
			if errMsg != nil {
				// An error occurred: emit as a proper SSE error event
				errorBytes, _ := json.Marshal(h.toClaudeError(errMsg))
				_, _ = writer.WriteString("event: error\n")
				_, _ = writer.WriteString("data: ")
				_, _ = writer.Write(errorBytes)
				_, _ = writer.WriteString("\n\n")
				_ = writer.Flush()
			}
			var execErr error
			if errMsg != nil {
				execErr = errMsg.Error
			}
			cancel(execErr)
			return
		}
	}
}

type claudeErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type claudeErrorResponse struct {
	Type  string            `json:"type"`
	Error claudeErrorDetail `json:"error"`
}

func (h *ClaudeCodeAPIHandler) toClaudeError(msg *interfaces.ErrorMessage) claudeErrorResponse {
	return claudeErrorResponse{
		Type: "error",
		Error: claudeErrorDetail{
			Type:    "api_error",
			Message: msg.Error.Error(),
		},
	}
}
