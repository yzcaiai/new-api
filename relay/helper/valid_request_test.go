package helper_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetAndValidateGeminiRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	newContext := func(body string) *gin.Context {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-3.7-flash:generateContent", bytes.NewBufferString(body))
		c.Request.Header.Set("Content-Type", "application/json")
		return c
	}

	t.Run("valid native request with thinkingLevel is accepted", func(t *testing.T) {
		body := `{
			"contents": [{"parts": [{"text": "Hello"}]}],
			"generationConfig": {
				"thinkingConfig": {
					"thinkingLevel": "high",
					"includeThoughts": true
				}
			}
		}`
		c := newContext(body)
		req, err := helper.GetAndValidateGeminiRequest(c)
		require.NoError(t, err)
		require.NotNil(t, req)
		require.NotNil(t, req.GenerationConfig.ThinkingConfig)
		assert.Equal(t, "high", req.GenerationConfig.ThinkingConfig.ThinkingLevel)
		assert.True(t, req.GenerationConfig.ThinkingConfig.IncludeThoughts)
		assert.Nil(t, req.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("valid native request with thinkingBudget is accepted", func(t *testing.T) {
		body := `{
			"contents": [{"parts": [{"text": "Hello"}]}],
			"generationConfig": {
				"thinkingConfig": {
					"thinkingBudget": 2048,
					"includeThoughts": true
				}
			}
		}`
		c := newContext(body)
		req, err := helper.GetAndValidateGeminiRequest(c)
		require.NoError(t, err)
		require.NotNil(t, req)
		require.NotNil(t, req.GenerationConfig.ThinkingConfig)
		assert.NotNil(t, req.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, 2048, *req.GenerationConfig.ThinkingConfig.ThinkingBudget)
		assert.Empty(t, req.GenerationConfig.ThinkingConfig.ThinkingLevel)
	})

	t.Run("native request with both thinkingBudget and thinkingLevel is rejected", func(t *testing.T) {
		body := `{
			"contents": [{"parts": [{"text": "Hello"}]}],
			"generationConfig": {
				"thinkingConfig": {
					"thinkingBudget": 2048,
					"thinkingLevel": "high"
				}
			}
		}`
		c := newContext(body)
		_, err := helper.GetAndValidateGeminiRequest(c)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "thinkingConfig cannot contain both thinkingBudget and thinkingLevel")
	})

	t.Run("native request with negative thinkingBudget is rejected", func(t *testing.T) {
		body := `{
			"contents": [{"parts": [{"text": "Hello"}]}],
			"generationConfig": {
				"thinkingConfig": {
					"thinkingBudget": -1
				}
			}
		}`
		c := newContext(body)
		_, err := helper.GetAndValidateGeminiRequest(c)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "thinkingBudget cannot be negative")
	})
}
