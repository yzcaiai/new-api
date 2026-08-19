package gemini_test

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/internal/shared/gemini"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApplyThinkingConfig_Gemini3And25(t *testing.T) {
	t.Run("gemini-3.7-flash maps reasoning_effort to thinkingLevel without budget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.7-flash",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}
		oaiReq := dto.GeneralOpenAIRequest{
			Model:           "gemini-3.7-flash",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, "high", meta.GetReasoningEffort())
	})

	t.Run("gemini-3.1-pro with effort none disables thinking", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.1-pro-preview",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}
		oaiReq := dto.GeneralOpenAIRequest{
			Model:           "gemini-3.1-pro-preview",
			ReasoningEffort: "none",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.False(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, 0, *geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-2.5-pro maps reasoning_effort to thinkingBudget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-2.5-pro",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}
		oaiReq := dto.GeneralOpenAIRequest{
			Model:           "gemini-2.5-pro",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Empty(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Equal(t, "high", meta.GetReasoningEffort())
	})
}
