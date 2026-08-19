package claudemessages_test

import (
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	claudemessages "github.com/QuantumNous/new-api/relaykit/relayconvert/internal/claude_messages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClaudeMessagesRequestToOpenAIChat_ReasoningEffort(t *testing.T) {
	t.Run("converts output_config effort to reasoning_effort on standard channel", func(t *testing.T) {
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}
		claudeReq := dto.ClaudeRequest{
			Model:        "deepseek-v4-pro",
			OutputConfig: json.RawMessage(`{"effort":"xhigh"}`),
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiReq, err := claudemessages.ClaudeMessagesRequestToOpenAIChat(claudeReq, meta)
		require.NoError(t, err)
		assert.Equal(t, "xhigh", oaiReq.ReasoningEffort)
		assert.Equal(t, "xhigh", meta.GetReasoningEffort())
	})

	t.Run("converts adaptive thinking to high reasoning_effort when effort unset", func(t *testing.T) {
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}
		claudeReq := dto.ClaudeRequest{
			Model:    "deepseek-v4-pro",
			Thinking: &dto.Thinking{Type: "adaptive"},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiReq, err := claudemessages.ClaudeMessagesRequestToOpenAIChat(claudeReq, meta)
		require.NoError(t, err)
		assert.Equal(t, "high", oaiReq.ReasoningEffort)
		assert.Equal(t, "high", meta.GetReasoningEffort())
	})

	t.Run("converts budget_tokens to appropriate reasoning_effort on standard channel (aligned with CCS)", func(t *testing.T) {
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}
		budgetHigh := 10240
		claudeReqHigh := dto.ClaudeRequest{
			Model:    "deepseek-v4-pro",
			Thinking: &dto.Thinking{Type: "enabled", BudgetTokens: &budgetHigh},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiReqHigh, err := claudemessages.ClaudeMessagesRequestToOpenAIChat(claudeReqHigh, meta)
		require.NoError(t, err)
		assert.Equal(t, "high", oaiReqHigh.ReasoningEffort)

		budgetMed := 2048
		claudeReqMed := dto.ClaudeRequest{
			Model:    "deepseek-v4-pro",
			Thinking: &dto.Thinking{Type: "enabled", BudgetTokens: &budgetMed},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiReqMed, err := claudemessages.ClaudeMessagesRequestToOpenAIChat(claudeReqMed, meta)
		require.NoError(t, err)
		assert.Equal(t, "medium", oaiReqMed.ReasoningEffort)
	})

	t.Run("converts thinking type disabled to reasoning_effort none", func(t *testing.T) {
		meta := &convmeta.Values{
			Options: &convmeta.Options{},
		}
		claudeReq := dto.ClaudeRequest{
			Model:    "deepseek-v4-pro",
			Thinking: &dto.Thinking{Type: "disabled"},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiReq, err := claudemessages.ClaudeMessagesRequestToOpenAIChat(claudeReq, meta)
		require.NoError(t, err)
		assert.Equal(t, "none", oaiReq.ReasoningEffort)
		assert.Equal(t, "none", meta.GetReasoningEffort())
	})

	t.Run("retains OpenRouter dialect fields", func(t *testing.T) {
		meta := &convmeta.Values{
			Options: &convmeta.Options{
				OpenRouterDialect: true,
			},
		}
		claudeReq := dto.ClaudeRequest{
			Model:        "anthropic/claude-3.7-sonnet",
			OutputConfig: json.RawMessage(`{"effort":"medium"}`),
			Thinking:     &dto.Thinking{Type: "adaptive"},
			Messages: []dto.ClaudeMessage{
				{Role: "user", Content: "Hello"},
			},
		}

		oaiReq, err := claudemessages.ClaudeMessagesRequestToOpenAIChat(claudeReq, meta)
		require.NoError(t, err)
		assert.Equal(t, "medium", oaiReq.ReasoningEffort)
		assert.NotEmpty(t, oaiReq.Verbosity)
		assert.NotEmpty(t, oaiReq.Reasoning)
	})
}
