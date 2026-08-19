package relay

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/stretchr/testify/assert"
)

func TestIsNoThinkingRequest(t *testing.T) {
	t.Run("gemini 2.5 thinkingBudget 0 is recognized as no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingBudget: kitutil.GetPointer(0),
				},
			},
		}
		assert.True(t, isNoThinkingRequest("gemini-2.5-pro", req))
		assert.True(t, isNoThinkingRequest("gemini-2.5-flash", req))
		assert.True(t, isNoThinkingRequest("gemini-2.5-flash-lite", req))
		assert.True(t, isNoThinkingRequest("google/gemini-2.5-pro", req))
	})

	t.Run("non-gemini-2.5 models with thinkingBudget 0 are NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingBudget: kitutil.GetPointer(0),
				},
			},
		}
		// Gemini 2.0 / 1.x / unknown / other mapped models
		assert.False(t, isNoThinkingRequest("gemini-2.0-flash", req))
		assert.False(t, isNoThinkingRequest("gemini-2.0-pro-exp-02-05", req))
		assert.False(t, isNoThinkingRequest("gemini-1.5-pro", req))
		assert.False(t, isNoThinkingRequest("gemini-1.5-flash", req))
		assert.False(t, isNoThinkingRequest("custom-gemini-channel-model", req))
		// Gemini 3 models
		assert.False(t, isNoThinkingRequest("gemini-3.7-flash", req))
		assert.False(t, isNoThinkingRequest("gemini-3.1-pro-preview", req))
		assert.False(t, isNoThinkingRequest("gemini-3-pro-preview", req))
		assert.False(t, isNoThinkingRequest("gemini-3-flash-preview", req))
	})

	t.Run("thinkingLevel high without includeThoughts is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingLevel: "high",
				},
			},
		}
		assert.False(t, isNoThinkingRequest("gemini-3.7-flash", req))
	})

	t.Run("thinkingLevel high with includeThoughts false is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingLevel:   "high",
					IncludeThoughts: false,
				},
			},
		}
		assert.False(t, isNoThinkingRequest("gemini-3.7-flash", req))
	})

	t.Run("thinkingLevel minimal is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingLevel:   "minimal",
					IncludeThoughts: true,
				},
			},
		}
		assert.False(t, isNoThinkingRequest("gemini-3.7-flash", req))
	})

	t.Run("positive thinkingBudget is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingBudget:  kitutil.GetPointer(2048),
					IncludeThoughts: true,
				},
			},
		}
		assert.False(t, isNoThinkingRequest("gemini-2.5-pro", req))
	})

	t.Run("nil thinkingConfig is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}
		assert.False(t, isNoThinkingRequest("gemini-2.5-pro", req))
	})
}
