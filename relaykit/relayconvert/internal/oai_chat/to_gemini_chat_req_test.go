package oaichat_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	oaichat "github.com/QuantumNous/new-api/relaykit/relayconvert/internal/oai_chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIChatRequestToGeminiGenerateContent_ThinkingConfigValidation(t *testing.T) {
	ctx := context.Background()

	t.Run("rejects request with both thinking_budget and thinking_level in extra_body", func(t *testing.T) {
		extraBody := map[string]interface{}{
			"google": map[string]interface{}{
				"thinking_config": map[string]interface{}{
					"thinking_budget": 2048,
					"thinking_level":  "high",
				},
			},
		}
		extraBodyBytes, _ := json.Marshal(extraBody)

		req := dto.GeneralOpenAIRequest{
			Model:     "gemini-3.1-pro-preview",
			ExtraBody: extraBodyBytes,
			Messages: []dto.Message{
				{Role: "user", Content: "hello"},
			},
		}
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}

		_, err := oaichat.OpenAIChatRequestToGeminiGenerateContent(ctx, req, meta)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cannot contain both thinking_budget and thinking_level")
	})

	t.Run("accepts valid thinking_level in extra_body", func(t *testing.T) {
		extraBody := map[string]interface{}{
			"google": map[string]interface{}{
				"thinking_config": map[string]interface{}{
					"thinking_level": "low",
				},
			},
		}
		extraBodyBytes, _ := json.Marshal(extraBody)

		req := dto.GeneralOpenAIRequest{
			Model:     "gemini-3.7-flash",
			ExtraBody: extraBodyBytes,
			Messages: []dto.Message{
				{Role: "user", Content: "hello"},
			},
		}
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}

		geminiReq, err := oaichat.OpenAIChatRequestToGeminiGenerateContent(ctx, req, meta)
		require.NoError(t, err)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("rejects thinkingLevel camelCase in extra_body", func(t *testing.T) {
		extraBody := map[string]interface{}{
			"google": map[string]interface{}{
				"thinking_config": map[string]interface{}{
					"thinkingLevel": "high",
				},
			},
		}
		extraBodyBytes, _ := json.Marshal(extraBody)

		req := dto.GeneralOpenAIRequest{
			Model:     "gemini-3.7-flash",
			ExtraBody: extraBodyBytes,
			Messages: []dto.Message{
				{Role: "user", Content: "hello"},
			},
		}
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}

		_, err := oaichat.OpenAIChatRequestToGeminiGenerateContent(ctx, req, meta)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "extra_body.google.thinking_config.thinkingLevel is not supported")
	})
}
