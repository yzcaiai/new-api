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

	t.Run("gemini-3.1-pro with effort none/off fallbacks to low level and logs effective low", func(t *testing.T) {
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
		// Effective reasoning effort is recorded as low
		assert.Equal(t, "low", meta.GetReasoningEffort())
	})

	t.Run("explicit effort off overrides -thinking suffix and logs effective low", func(t *testing.T) {
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
		assert.Equal(t, "low", meta.GetReasoningEffort())
	})

	t.Run("profile coverage: gemini-3-pro-preview only supports low/high (medium safely fallbacks to default high)", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-pro-preview",
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
			Model:           "gemini-3-pro-preview",
			ReasoningEffort: "medium",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		// gemini-3-pro-preview only supports low/high; medium falls back to default high
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, "high", meta.GetReasoningEffort())
	})

	t.Run("profile coverage: gemini-3-pro-preview with low effort resolves to low", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-pro-preview",
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
			Model:           "gemini-3-pro-preview",
			ReasoningEffort: "low",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, "low", meta.GetReasoningEffort())
	})

	t.Run("profile coverage: gemini-3.1-flash-lite-preview supports low/medium/high", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.1-flash-lite-preview",
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
			Model:           "gemini-3.1-flash-lite-preview",
			ReasoningEffort: "medium",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "medium", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, "medium", meta.GetReasoningEffort())
	})

	t.Run("profile coverage: gemini-3.1-flash-image does not inject thinking config", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3.1-flash-image-preview",
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
			Model:           "gemini-3.1-flash-image-preview",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig)
	})

	t.Run("profile coverage: gemini-3-pro-image with -thinking-2048 does not inject thinking config", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-pro-image-thinking-2048",
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
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig)
	})

	t.Run("profile coverage: gemini-3-future-unknown with -thinking-2048 does not inject thinking config", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-future-unknown-variant-thinking-2048",
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
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig)
	})

	t.Run("profile coverage: gemini-3-pro-future-unknown does not match gemini-3-pro-preview and does not inject", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-pro-future-unknown",
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
			Model:           "gemini-3-pro-future-unknown",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig)
	})

	t.Run("profile coverage: gemini-3-flash-preview supports thinking", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-flash-preview",
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
			Model:           "gemini-3-flash-preview",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, "high", meta.GetReasoningEffort())
	})

	t.Run("profile coverage: gemini-3-pro-image does not inject thinking config", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-pro-image",
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
			Model:           "gemini-3-pro-image",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig)
	})

	t.Run("profile coverage: unknown gemini 3 model does not guess thinking config", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName: "gemini-3-future-unknown-variant",
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
			Model:           "gemini-3-future-unknown-variant",
			ReasoningEffort: "high",
		}

		gemini.ApplyThinkingConfig(geminiReq, meta, oaiReq)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig)
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
