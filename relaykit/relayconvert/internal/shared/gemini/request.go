package gemini

import (
	"strconv"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/convmeta"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/reasoning"
)

var SupportedMimeTypes = map[string]bool{
	"application/pdf": true,
	"audio/mpeg":      true,
	"audio/mp3":       true,
	"audio/wav":       true,
	"image/png":       true,
	"image/jpeg":      true,
	"image/jpg":       true,
	"image/webp":      true,
	"image/heic":      true,
	"image/heif":      true,
	"text/plain":      true,
	"video/mov":       true,
	"video/mpeg":      true,
	"video/mp4":       true,
	"video/mpg":       true,
	"video/avi":       true,
	"video/wmv":       true,
	"video/mpegps":    true,
	"video/flv":       true,
}

var SafetySettingCategories = []string{
	"HARM_CATEGORY_HARASSMENT",
	"HARM_CATEGORY_HATE_SPEECH",
	"HARM_CATEGORY_SEXUALLY_EXPLICIT",
	"HARM_CATEGORY_DANGEROUS_CONTENT",
}

const ThoughtSignatureBypassValue = "context_engineering_is_the_way_to_go"

const (
	pro25MinBudget       = 128
	pro25MaxBudget       = 32768
	flash25MaxBudget     = 24576
	flash25LiteMinBudget = 512
	flash25LiteMaxBudget = 24576
)

func ShouldAttachThoughtSignature(opts *convmeta.Options) bool {
	return opts != nil && opts.Gemini.FunctionCallThoughtSignatureEnabled
}

func AttachThoughtSignatureBypass(opts *convmeta.Options, part *dto.GeminiPart) bool {
	if part == nil || len(part.ThoughtSignature) > 0 || !ShouldAttachThoughtSignature(opts) {
		return false
	}
	part.ThoughtSignature = []byte(strconv.Quote(ThoughtSignatureBypassValue))
	return true
}

func AttachFunctionCallThoughtSignature(opts *convmeta.Options, part *dto.GeminiPart) bool {
	if part == nil || !HasFunctionCallContent(part.FunctionCall) {
		return false
	}
	return AttachThoughtSignatureBypass(opts, part)
}

func AttachFirstTextThoughtSignature(opts *convmeta.Options, parts []dto.GeminiPart) bool {
	if !ShouldAttachThoughtSignature(opts) {
		return false
	}
	for i := range parts {
		if parts[i].Text != "" && len(parts[i].ThoughtSignature) == 0 {
			parts[i].ThoughtSignature = []byte(strconv.Quote(ThoughtSignatureBypassValue))
			return true
		}
	}
	return false
}

func ApplyThinkingConfig(geminiRequest *dto.GeminiChatRequest, info convmeta.Meta, oaiRequest ...dto.GeneralOpenAIRequest) {
	opts := convmeta.OptionsOf(info)
	if geminiRequest == nil || info == nil || !opts.Gemini.ThinkingAdapterEnabled {
		return
	}

	modelName := convmeta.UpstreamModelName(info)
	if modelName == "" {
		if values, ok := info.(*convmeta.Values); ok && values.UpstreamModelName != "" {
			modelName = values.UpstreamModelName
		} else if info != nil && info.GetOriginModelName() != "" {
			modelName = info.GetOriginModelName()
		} else if len(oaiRequest) > 0 && oaiRequest[0].Model != "" {
			modelName = oaiRequest[0].Model
		}
	}
	isNew25Pro := isNew25ProModel(modelName)
	isGem3 := isGemini3Model(modelName)

	var reqEffort string
	if len(oaiRequest) > 0 && oaiRequest[0].ReasoningEffort != "" {
		reqEffort = oaiRequest[0].ReasoningEffort
	} else if info != nil && info.GetReasoningEffort() != "" {
		reqEffort = info.GetReasoningEffort()
	}

	if strings.Contains(modelName, "-thinking-") {
		parts := strings.SplitN(modelName, "-thinking-", 2)
		if len(parts) == 2 && parts[1] != "" {
			if budgetTokens, err := strconv.Atoi(parts[1]); err == nil {
				if isGem3 {
					level := budgetToGemini3ThinkingLevel(modelName, budgetTokens)
					geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
						IncludeThoughts: true,
						ThinkingLevel:   level,
					}
					info.SetReasoningEffort(level)
				} else {
					clampedBudget := clampThinkingBudget(modelName, budgetTokens)
					geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
						ThinkingBudget:  kitutil.GetPointer(clampedBudget),
						IncludeThoughts: true,
					}
				}
			}
		}
	} else if strings.HasSuffix(modelName, "-thinking") {
		unsupportedModels := []string{
			"gemini-2.5-pro-preview-05-06",
			"gemini-2.5-pro-preview-03-25",
		}
		isUnsupported := false
		for _, unsupportedModel := range unsupportedModels {
			if strings.HasPrefix(modelName, unsupportedModel) {
				isUnsupported = true
				break
			}
		}

		if isUnsupported {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
			}
		} else if isGem3 {
			level := "high"
			if reqEffort != "" {
				level = resolveGemini3ThinkingLevel(modelName, reqEffort)
			}
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingLevel:   level,
			}
			info.SetReasoningEffort(level)
		} else {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
			}
			if geminiRequest.GenerationConfig.MaxOutputTokens != nil && *geminiRequest.GenerationConfig.MaxOutputTokens > 0 {
				budgetTokens := opts.Gemini.ThinkingAdapterBudgetTokensPercentage * float64(*geminiRequest.GenerationConfig.MaxOutputTokens)
				clampedBudget := clampThinkingBudget(modelName, int(budgetTokens))
				geminiRequest.GenerationConfig.ThinkingConfig.ThinkingBudget = kitutil.GetPointer(clampedBudget)
			} else if reqEffort != "" {
				geminiRequest.GenerationConfig.ThinkingConfig.ThinkingBudget = kitutil.GetPointer(clampThinkingBudgetByEffort(modelName, reqEffort))
				info.SetReasoningEffort(reqEffort)
			}
		}
	} else if strings.HasSuffix(modelName, "-nothinking") {
		if isGem3 {
			geminiRequest.GenerationConfig.ThinkingConfig = resolveGemini3NoThinkingConfig(modelName)
			info.SetReasoningEffort("none")
		} else if !isNew25Pro {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				ThinkingBudget: kitutil.GetPointer(0),
			}
		}
	} else if _, level, ok := reasoning.TrimEffortSuffix(modelName); ok && level != "" {
		if isGem3 {
			resolvedLevel := resolveGemini3ThinkingLevel(modelName, level)
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingLevel:   resolvedLevel,
			}
			info.SetReasoningEffort(resolvedLevel)
		} else {
			// Preserve baseline behavior for Gemini 2.5 suffix models (uses ThinkingLevel)
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingLevel:   level,
			}
			info.SetReasoningEffort(level)
		}
	} else if reqEffort != "" {
		if reqEffort == "none" || reqEffort == "off" {
			if isGem3 {
				geminiRequest.GenerationConfig.ThinkingConfig = resolveGemini3NoThinkingConfig(modelName)
				info.SetReasoningEffort("none")
			} else if !isNew25Pro {
				geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
					ThinkingBudget: kitutil.GetPointer(0),
				}
			}
		} else if isGem3 {
			resolvedLevel := resolveGemini3ThinkingLevel(modelName, reqEffort)
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingLevel:   resolvedLevel,
			}
			info.SetReasoningEffort(resolvedLevel)
		} else {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingBudget:  kitutil.GetPointer(clampThinkingBudgetByEffort(modelName, reqEffort)),
			}
			info.SetReasoningEffort(reqEffort)
		}
	}
}

func ParseStopSequences(stop any) []string {
	if stop == nil {
		return nil
	}

	switch v := stop.(type) {
	case string:
		if v != "" {
			return []string{v}
		}
	case []string:
		return v
	case []interface{}:
		sequences := make([]string, 0, len(v))
		for _, item := range v {
			if str, ok := item.(string); ok && str != "" {
				sequences = append(sequences, str)
			}
		}
		return sequences
	}
	return nil
}

func HasFunctionCallContent(call *dto.FunctionCall) bool {
	if call == nil {
		return false
	}
	if strings.TrimSpace(call.FunctionName) != "" {
		return true
	}

	switch v := call.Arguments.(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(v) != ""
	case map[string]interface{}:
		return len(v) > 0
	case []interface{}:
		return len(v) > 0
	default:
		return true
	}
}

func SupportedMimeTypesList() []string {
	keys := make([]string, 0, len(SupportedMimeTypes))
	for key := range SupportedMimeTypes {
		keys = append(keys, key)
	}
	return keys
}

func isGemini3Model(modelName string) bool {
	lower := strings.ToLower(modelName)
	return strings.HasPrefix(lower, "gemini-3") ||
		strings.Contains(lower, "gemini-3.0") ||
		strings.Contains(lower, "gemini-3.1") ||
		strings.Contains(lower, "gemini-3.5") ||
		strings.Contains(lower, "gemini-3.7")
}

// resolveGemini3ThinkingLevel maps arbitrary input effort/level to a valid Gemini 3 thinking level.
// Supported levels for Gemini 3 models: "low", "medium", "high", "minimal".
// Claude's "xhigh" / "max" are normalized to "high".
// If the target model does not support "minimal" (e.g. gemini-3.7-flash), it downgrades to "low".
func resolveGemini3ThinkingLevel(modelName string, effort string) string {
	clean := strings.ToLower(strings.TrimSpace(effort))
	var targetLevel string
	switch clean {
	case "xhigh", "max", "high":
		targetLevel = "high"
	case "medium":
		targetLevel = "medium"
	case "low":
		targetLevel = "low"
	case "minimal":
		targetLevel = "minimal"
	default:
		targetLevel = "high"
	}

	lowerModel := strings.ToLower(modelName)
	// gemini-3.7-flash does not support minimal; downgrade to low
	if strings.Contains(lowerModel, "gemini-3.7-flash") && targetLevel == "minimal" {
		targetLevel = "low"
	}
	return targetLevel
}

// budgetToGemini3ThinkingLevel converts numeric thinking budget to Gemini 3 thinking level (aligned with CCS: >=8192 => high, else medium).
func budgetToGemini3ThinkingLevel(modelName string, budgetTokens int) string {
	if budgetTokens >= 8192 {
		return resolveGemini3ThinkingLevel(modelName, "high")
	}
	return resolveGemini3ThinkingLevel(modelName, "medium")
}

// resolveGemini3NoThinkingConfig determines the Gemini 3 config for disabled/none/off/-nothinking requests.
// For models supporting minimal (e.g. gemini-3.1-pro), maps to thinkingLevel="minimal" with includeThoughts=false.
// For other models, sets thinkingLevel="low" with includeThoughts=false.
func resolveGemini3NoThinkingConfig(modelName string) *dto.GeminiThinkingConfig {
	lowerModel := strings.ToLower(modelName)
	if strings.Contains(lowerModel, "gemini-3.7-flash") {
		return &dto.GeminiThinkingConfig{
			IncludeThoughts: false,
			ThinkingLevel:   "low",
		}
	}
	return &dto.GeminiThinkingConfig{
		IncludeThoughts: false,
		ThinkingLevel:   "minimal",
	}
}

func isNew25ProModel(modelName string) bool {
	return strings.HasPrefix(modelName, "gemini-2.5-pro") &&
		!strings.HasPrefix(modelName, "gemini-2.5-pro-preview-05-06") &&
		!strings.HasPrefix(modelName, "gemini-2.5-pro-preview-03-25")
}

func is25FlashLiteModel(modelName string) bool {
	return strings.HasPrefix(modelName, "gemini-2.5-flash-lite")
}

func clampThinkingBudget(modelName string, budget int) int {
	isNew25Pro := isNew25ProModel(modelName)
	is25FlashLite := is25FlashLiteModel(modelName)

	if is25FlashLite {
		if budget < flash25LiteMinBudget {
			return flash25LiteMinBudget
		}
		if budget > flash25LiteMaxBudget {
			return flash25LiteMaxBudget
		}
	} else if isNew25Pro {
		if budget < pro25MinBudget {
			return pro25MinBudget
		}
		if budget > pro25MaxBudget {
			return pro25MaxBudget
		}
	} else {
		if budget < 0 {
			return 0
		}
		if budget > flash25MaxBudget {
			return flash25MaxBudget
		}
	}
	return budget
}

func clampThinkingBudgetByEffort(modelName string, effort string) int {
	isNew25Pro := isNew25ProModel(modelName)
	is25FlashLite := is25FlashLiteModel(modelName)

	maxBudget := 0
	if is25FlashLite {
		maxBudget = flash25LiteMaxBudget
	}
	if isNew25Pro {
		maxBudget = pro25MaxBudget
	} else {
		maxBudget = flash25MaxBudget
	}
	switch effort {
	case "high":
		maxBudget = maxBudget * 80 / 100
	case "medium":
		maxBudget = maxBudget * 50 / 100
	case "low":
		maxBudget = maxBudget * 20 / 100
	case "minimal":
		maxBudget = maxBudget * 5 / 100
	}
	return clampThinkingBudget(modelName, maxBudget)
}
