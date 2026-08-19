package relay

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/stretchr/testify/assert"
)

func TestIsNoThinkingRequest(t *testing.T) {
	t.Run("thinkingBudget 0 is recognized as no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingBudget: kitutil.GetPointer(0),
				},
			},
		}
		assert.True(t, isNoThinkingRequest(req))
	})

	t.Run("thinkingLevel high without includeThoughts is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{
				ThinkingConfig: &dto.GeminiThinkingConfig{
					ThinkingLevel: "high",
				},
			},
		}
		assert.False(t, isNoThinkingRequest(req))
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
		assert.False(t, isNoThinkingRequest(req))
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
		assert.False(t, isNoThinkingRequest(req))
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
		assert.False(t, isNoThinkingRequest(req))
	})

	t.Run("nil thinkingConfig is NOT no-thinking request", func(t *testing.T) {
		req := &dto.GeminiChatRequest{
			GenerationConfig: dto.GeminiChatGenerationConfig{},
		}
		assert.False(t, isNoThinkingRequest(req))
	})
}
