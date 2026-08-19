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

	t.Run("gemini-3.1-pro with effort xhigh/max resolves to high thinkingLevel without budget", func(t *testing.T) {
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
			ReasoningEffort: "xhigh",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, "high", meta.GetReasoningEffort())
	})

	t.Run("gemini-3.1-pro with minimal downgrades to low", func(t *testing.T) {
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
			ReasoningEffort: "minimal",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-3.7-flash with minimal downgrades to low", func(t *testing.T) {
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
			ReasoningEffort: "minimal",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-3 -thinking-8192 converts to thinkingLevel high without budget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.1-pro-preview-thinking-8192",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}

		gemini.ApplyThinkingConfig(geminiReq, meta)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-3 -thinking-2048 converts to thinkingLevel medium without budget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.7-flash-thinking-2048",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}

		gemini.ApplyThinkingConfig(geminiReq, meta)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "medium", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-3 -nothinking sets low level without includeThoughts and without budget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.1-pro-preview-nothinking",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}

		gemini.ApplyThinkingConfig(geminiReq, meta)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.False(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-3.1-pro with effort none/off (with whitespace & case) disables thinking using low level", func(t *testing.T) {
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
			ReasoningEffort: " NONE ",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.False(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("explicit effort none overrides -thinking suffix", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.7-flash-thinking",
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
			Model:           "gemini-3.7-flash-thinking",
			ReasoningEffort: "off",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.False(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-2.5-pro suffix regression check", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-2.5-pro-high",
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}
		geminiReq := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}

		gemini.ApplyThinkingConfig(geminiReq, meta)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
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
