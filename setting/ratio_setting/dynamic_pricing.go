package ratio_setting

const (
	TextPricingModeStandard = "standard"
	TextPricingModeBatch    = "batch"
	TextPricingModeFlex     = "flex"
	TextPricingModePriority = "priority"
)

type InputThresholdPricingRule struct {
	Threshold                  int
	InputPriceMultiplier       float64
	OutputPriceMultiplier      float64
	CachedInputPriceMultiplier *float64
	ModeModelRatioMultiplier   map[string]float64
	LongContextModes           map[string]struct{}
}

var (
	longContextDoubleMultiplier    = 2.0
	textInputThresholdPricingRules = map[string]InputThresholdPricingRule{
		"gpt-5.4": {
			Threshold:                  272000,
			InputPriceMultiplier:       2.0,
			OutputPriceMultiplier:      1.5,
			CachedInputPriceMultiplier: &longContextDoubleMultiplier,
			ModeModelRatioMultiplier: map[string]float64{
				TextPricingModeStandard: 1.0,
				TextPricingModeBatch:    0.5,
				TextPricingModeFlex:     0.5,
			},
			LongContextModes: map[string]struct{}{
				TextPricingModeStandard: {},
				TextPricingModeBatch:    {},
				TextPricingModeFlex:     {},
			},
		},
		"gpt-5.4-pro": {
			Threshold:             272000,
			InputPriceMultiplier:  2.0,
			OutputPriceMultiplier: 1.5,
			ModeModelRatioMultiplier: map[string]float64{
				TextPricingModeStandard: 1.0,
				TextPricingModeBatch:    0.5,
				TextPricingModeFlex:     0.5,
			},
			LongContextModes: map[string]struct{}{
				TextPricingModeStandard: {},
				TextPricingModeBatch:    {},
				TextPricingModeFlex:     {},
			},
		},
	}
)

func normalizeTextPricingMode(mode string) string {
	switch mode {
	case TextPricingModeBatch, TextPricingModeFlex, TextPricingModePriority:
		return mode
	default:
		return TextPricingModeStandard
	}
}

func HasTextInputThresholdPricingRule(modelName string) bool {
	_, ok := textInputThresholdPricingRules[FormatMatchingModelName(modelName)]
	return ok
}

func GetTextInputThresholdPricingRule(modelName string) (InputThresholdPricingRule, bool) {
	rule, ok := textInputThresholdPricingRules[FormatMatchingModelName(modelName)]
	return rule, ok
}

func ResolveTextPricingRatios(modelName string, inputTokens int, processingMode string, modelRatio, completionRatio, cacheRatio float64) (float64, float64, float64, bool) {
	rule, ok := textInputThresholdPricingRules[FormatMatchingModelName(modelName)]
	if !ok {
		return modelRatio, completionRatio, cacheRatio, false
	}

	mode := normalizeTextPricingMode(processingMode)
	if multiplier, ok := rule.ModeModelRatioMultiplier[mode]; ok {
		modelRatio *= multiplier
	}

	if inputTokens <= rule.Threshold {
		return modelRatio, completionRatio, cacheRatio, false
	}
	if _, ok := rule.LongContextModes[mode]; !ok {
		return modelRatio, completionRatio, cacheRatio, false
	}

	modelRatio *= rule.InputPriceMultiplier
	if completionRatio != 0 && rule.InputPriceMultiplier != 0 {
		completionRatio *= rule.OutputPriceMultiplier / rule.InputPriceMultiplier
	}
	if rule.CachedInputPriceMultiplier != nil && rule.InputPriceMultiplier != 0 {
		cacheRatio *= *rule.CachedInputPriceMultiplier / rule.InputPriceMultiplier
	}
	return modelRatio, completionRatio, cacheRatio, true
}
