package relayconvert_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClaudeToOpenAIToGeminiE2EChain(t *testing.T) {
	ctx := context.Background()

	t.Run("Claude adaptive with effort xhigh converts through OpenAI to Gemini 3.1 Pro wire JSON", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName:   "gemini-3.1-pro-preview",
			ChannelMetaAttached: true,
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}

		claudeReq := &dto.ClaudeRequest{
			Model:        "claude-opus-4-8",
			OutputConfig: json.RawMessage(`{"effort":"xhigh"}`),
			Thinking:     &dto.Thinking{Type: "adaptive"},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Write an algorithm"},
			},
		}

		// 1. Claude -> OpenAI Chat
		oaiResult, err := relayconvert.ConvertRequest(ctx, meta, types.RelayFormatOpenAI, claudeReq)
		require.NoError(t, err)
		oaiReq, ok := oaiResult.Value.(*dto.GeneralOpenAIRequest)
		require.True(t, ok)
		assert.Equal(t, "xhigh", oaiReq.ReasoningEffort)

		// 2. OpenAI Chat -> Gemini Chat
		geminiResult, err := relayconvert.ConvertRequest(ctx, meta, types.RelayFormatGemini, oaiReq)
		require.NoError(t, err)
		geminiReq, ok := geminiResult.Value.(*dto.GeminiChatRequest)
		require.True(t, ok)

		// Assert struct fields
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.True(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "high", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)

		// Assert Wire JSON
		wireBytes, err := json.Marshal(geminiReq)
		require.NoError(t, err)
		wireJSON := string(wireBytes)
		assert.Contains(t, wireJSON, `"thinkingLevel":"high"`)
		assert.NotContains(t, wireJSON, "thinkingBudget")
	})

	t.Run("Claude thinking disabled converts through OpenAI to Gemini 3.1 Pro wire JSON as low without budget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName:   "gemini-3.1-pro-preview",
			ChannelMetaAttached: true,
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}

		claudeReq := &dto.ClaudeRequest{
			Model:    "claude-opus-4-8",
			Thinking: &dto.Thinking{Type: "disabled"},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		// 1. Claude -> OpenAI Chat
		oaiResult, err := relayconvert.ConvertRequest(ctx, meta, types.RelayFormatOpenAI, claudeReq)
		require.NoError(t, err)
		oaiReq, ok := oaiResult.Value.(*dto.GeneralOpenAIRequest)
		require.True(t, ok)
		assert.Equal(t, "none", oaiReq.ReasoningEffort)

		// 2. OpenAI Chat -> Gemini Chat
		geminiResult, err := relayconvert.ConvertRequest(ctx, meta, types.RelayFormatGemini, oaiReq)
		require.NoError(t, err)
		geminiReq, ok := geminiResult.Value.(*dto.GeminiChatRequest)
		require.True(t, ok)

		// Assert struct fields: Gemini 3.1 Pro lowest level is "low"
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.False(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)

		// Assert Wire JSON
		wireBytes, err := json.Marshal(geminiReq)
		require.NoError(t, err)
		wireJSON := string(wireBytes)
		assert.Contains(t, wireJSON, `"thinkingLevel":"low"`)
		assert.NotContains(t, wireJSON, "thinkingBudget")
	})

	t.Run("Claude thinking disabled converts through OpenAI to Gemini 3.7 Flash as low without budget", func(t *testing.T) {
		meta := &convmeta.Values{
			UpstreamModelName:   "gemini-3.7-flash",
			ChannelMetaAttached: true,
			Options: &convmeta.Options{
				Gemini: convmeta.GeminiOptions{
					ThinkingAdapterEnabled: true,
				},
			},
		}

		claudeReq := &dto.ClaudeRequest{
			Model:    "claude-opus-4-8",
			Thinking: &dto.Thinking{Type: "disabled"},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiResult, err := relayconvert.ConvertRequest(ctx, meta, types.RelayFormatOpenAI, claudeReq)
		require.NoError(t, err)
		oaiReq, ok := oaiResult.Value.(*dto.GeneralOpenAIRequest)
		require.True(t, ok)

		geminiResult, err := relayconvert.ConvertRequest(ctx, meta, types.RelayFormatGemini, oaiReq)
		require.NoError(t, err)
		geminiReq, ok := geminiResult.Value.(*dto.GeminiChatRequest)
		require.True(t, ok)

		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig)
		assert.False(t, geminiReq.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, "low", geminiReq.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.Nil(t, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})
}
