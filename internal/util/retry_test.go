package util

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWithRetry_Success(t *testing.T) {
	ctx := context.Background()
	callCount := 0

	result, err := WithRetry(ctx, 3, "Test", func(ctx context.Context) (string, error) {
		callCount++
		return "success", nil
	})

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if result != "success" {
		t.Errorf("Expected 'success', got '%s'", result)
	}
	if callCount != 1 {
		t.Errorf("Expected 1 call, got %d", callCount)
	}
}

func TestWithRetry_AllFail(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	testErr := errors.New("test error")

	result, err := WithRetry(ctx, 2, "Test", func(ctx context.Context) (string, error) {
		callCount++
		return "", testErr
	})

	if err == nil {
		t.Fatal("Expected error, got nil")
	}
	if !errors.Is(err, testErr) {
		t.Errorf("Expected wrapped test error, got %v", err)
	}
	if callCount != 2 {
		t.Errorf("Expected 2 calls, got %d", callCount)
	}
	// Should return zero value
	if result != "" {
		t.Errorf("Expected empty string, got '%s'", result)
	}
}

func TestWithRetry_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	result, err := WithRetry(ctx, 5, "Test", func(ctx context.Context) (string, error) {
		cancel() // Cancel on first call
		return "", errors.New("should not reach here")
	})

	if err == nil {
		t.Fatal("Expected context error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
	// Should return zero value
	if result != "" {
		t.Errorf("Expected empty string, got '%s'", result)
	}
}

func TestWithRetry_RetriesWithDelay(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	start := time.Now()

	result, err := WithRetry(ctx, 3, "Test", func(ctx context.Context) (string, error) {
		callCount++
		if callCount < 3 {
			return "", errors.New("fail")
		}
		return "success", nil
	})

	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if result != "success" {
		t.Errorf("Expected 'success', got '%s'", result)
	}
	if callCount != 3 {
		t.Errorf("Expected 3 calls, got %d", callCount)
	}
	// Should have taken at least ~3 seconds (1+2 second delays)
	if elapsed < 3*time.Second {
		t.Errorf("Expected at least 3s delay, took %v", elapsed)
	}
}

func TestWithRetry_MaxRetries(t *testing.T) {
	ctx := context.Background()
	callCount := 0

	result, err := WithRetry(ctx, 2, "Test", func(ctx context.Context) (string, error) {
		callCount++
		return "", errors.New("always fail")
	})

	if err == nil {
		t.Fatal("Expected error, got nil")
	}
	if callCount != 2 {
		t.Errorf("Expected 2 calls, got %d", callCount)
	}
	if result != "" {
		t.Errorf("Expected zero value, got '%s'", result)
	}
}
