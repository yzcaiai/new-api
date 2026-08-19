package gemini

import (
	"errors"
	"math"

	"github.com/QuantumNous/new-api/relaykit/dto"
)

// ValidateThinkingConfig verifies that thinkingBudget and thinkingLevel are not both specified,
// and that thinkingConfig parameters are valid.
func ValidateThinkingConfig(tc *dto.GeminiThinkingConfig) error {
	if tc == nil {
		return nil
	}
	hasBudget := tc.ThinkingBudget != nil
	hasLevel := tc.ThinkingLevel != ""
	if hasBudget && hasLevel {
		return errors.New("thinking_config cannot contain both thinking_budget and thinking_level")
	}
	if hasBudget && *tc.ThinkingBudget < 0 {
		return errors.New("thinking_budget cannot be negative")
	}
	return nil
}

// ValidateRawGoogleThinkingConfig validates the raw map parsed from extra_body.google.thinking_config.
func ValidateRawGoogleThinkingConfig(thinkingConfig map[string]interface{}) (*dto.GeminiThinkingConfig, error) {
	if thinkingConfig == nil {
		return nil, nil
	}

	if _, hasErrorParam := thinkingConfig["thinkingBudget"]; hasErrorParam {
		return nil, errors.New("extra_body.google.thinking_config.thinkingBudget is not supported, use extra_body.google.thinking_config.thinking_budget instead")
	}
	if _, hasErrorParam := thinkingConfig["thinkingLevel"]; hasErrorParam {
		return nil, errors.New("extra_body.google.thinking_config.thinkingLevel is not supported, use extra_body.google.thinking_config.thinking_level instead")
	}

	_, hasBudget := thinkingConfig["thinking_budget"]
	_, hasLevel := thinkingConfig["thinking_level"]
	if hasBudget && hasLevel {
		return nil, errors.New("extra_body.google.thinking_config cannot contain both thinking_budget and thinking_level")
	}

	res := &dto.GeminiThinkingConfig{}
	var hasConfig bool

	if thinkingBudget, exists := thinkingConfig["thinking_budget"]; exists {
		switch v := thinkingBudget.(type) {
		case float64:
			if v != math.Trunc(v) {
				return nil, errors.New("extra_body.google.thinking_config.thinking_budget must be an integer")
			}
			budgetInt := int(v)
			if budgetInt < 0 {
				return nil, errors.New("extra_body.google.thinking_config.thinking_budget cannot be negative")
			}
			res.ThinkingBudget = &budgetInt
			res.IncludeThoughts = budgetInt > 0
			hasConfig = true
		case int:
			if v < 0 {
				return nil, errors.New("extra_body.google.thinking_config.thinking_budget cannot be negative")
			}
			res.ThinkingBudget = &v
			res.IncludeThoughts = v > 0
			hasConfig = true
		default:
			return nil, errors.New("extra_body.google.thinking_config.thinking_budget must be an integer")
		}
	}

	if includeThoughts, exists := thinkingConfig["include_thoughts"]; exists {
		if v, ok := includeThoughts.(bool); ok {
			res.IncludeThoughts = v
			hasConfig = true
		} else {
			return nil, errors.New("extra_body.google.thinking_config.include_thoughts must be a boolean")
		}
	}

	if thinkingLevel, exists := thinkingConfig["thinking_level"]; exists {
		if v, ok := thinkingLevel.(string); ok {
			res.ThinkingLevel = v
			hasConfig = true
		} else {
			return nil, errors.New("extra_body.google.thinking_config.thinking_level must be a string")
		}
	}

	if !hasConfig {
		return nil, nil
	}
	return res, nil
}
