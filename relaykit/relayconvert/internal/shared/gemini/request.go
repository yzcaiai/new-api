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
		if len(oaiRequest) > 0 && oaiRequest[0].Model != "" {
			modelName = oaiRequest[0].Model
		} else if info != nil && info.GetOriginModelName() != "" {
			modelName = info.GetOriginModelName()
		} else if info != nil && info.GetUpstreamModelName() != "" {
			modelName = info.GetUpstreamModelName()
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
	normalizedEffort := strings.ToLower(strings.TrimSpace(reqEffort))

	// 1. Explicit effort is "none" or "off": highest precedence over model name suffixes
	if normalizedEffort == "none" || normalizedEffort == "off" {
		if isGem3 {
			geminiRequest.GenerationConfig.ThinkingConfig = resolveGemini3NoThinkingConfig(modelName)
			info.SetReasoningEffort("none")
		} else if !isNew25Pro {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				ThinkingBudget: kitutil.GetPointer(0),
			}
			info.SetReasoningEffort("none")
		}
		return
	}

	// 2. Explicit numeric budget suffix: -thinking-N
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
				return
			}
		}
	}

	// 3. Generic -thinking suffix
	if strings.HasSuffix(modelName, "-thinking") {
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
			if normalizedEffort != "" {
				level = resolveGemini3ThinkingLevel(modelName, normalizedEffort)
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
			} else if normalizedEffort != "" {
				geminiRequest.GenerationConfig.ThinkingConfig.ThinkingBudget = kitutil.GetPointer(clampThinkingBudgetByEffort(modelName, normalizedEffort))
				info.SetReasoningEffort(normalizedEffort)
			}
		}
		return
	}

	// 4. Model suffix -nothinking
	if strings.HasSuffix(modelName, "-nothinking") {
		if isGem3 {
			geminiRequest.GenerationConfig.ThinkingConfig = resolveGemini3NoThinkingConfig(modelName)
			info.SetReasoningEffort("none")
		} else if !isNew25Pro {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				ThinkingBudget: kitutil.GetPointer(0),
			}
			info.SetReasoningEffort("none")
		}
		return
	}

	// 5. Model suffix with effort level (-high, -low, -medium, -minimal, -max, -xhigh)
	if _, level, ok := reasoning.TrimEffortSuffix(modelName); ok && level != "" {
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
		return
	}

	// 6. Request contains explicit reasoning effort
	if normalizedEffort != "" {
		if isGem3 {
			resolvedLevel := resolveGemini3ThinkingLevel(modelName, normalizedEffort)
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingLevel:   resolvedLevel,
			}
			info.SetReasoningEffort(resolvedLevel)
		} else {
			geminiRequest.GenerationConfig.ThinkingConfig = &dto.GeminiThinkingConfig{
				IncludeThoughts: true,
				ThinkingBudget:  kitutil.GetPointer(clampThinkingBudgetByEffort(modelName, normalizedEffort)),
			}
			info.SetReasoningEffort(normalizedEffort)
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

func cleanModelName(modelName string) string {
	s := strings.TrimSpace(modelName)
	if idx := strings.Index(s, ":"); idx != -1 {
		s = s[:idx]
	}
	if idx := strings.LastIndex(s, "/"); idx != -1 {
		s = s[idx+1:]
	}
	s = strings.ToLower(s)
	if strings.HasSuffix(s, "-nothinking") {
		s = strings.TrimSuffix(s, "-nothinking")
	} else if strings.HasSuffix(s, "-thinking") {
		s = strings.TrimSuffix(s, "-thinking")
	} else if strings.Contains(s, "-thinking-") {
		parts := strings.SplitN(s, "-thinking-", 2)
		s = parts[0]
	} else if base, _, ok := reasoning.TrimEffortSuffix(s); ok {
		s = base
	}
	return s
}

func isGemini3Model(modelName string) bool {
	base := cleanModelName(modelName)
	return strings.HasPrefix(base, "gemini-3") ||
		strings.Contains(base, "gemini-3.0") ||
		strings.Contains(base, "gemini-3.1") ||
		strings.Contains(base, "gemini-3.5") ||
		strings.Contains(base, "gemini-3.7")
}

type gemini3Capability struct {
	allowedLevels []string
	defaultLevel  string
	lowestLevel   string
}

func getGemini3Capability(modelName string) gemini3Capability {
	base := cleanModelName(modelName)
	switch {
	case strings.Contains(base, "gemini-3.7-flash"):
		return gemini3Capability{
			allowedLevels: []string{"low", "medium", "high"},
			defaultLevel:  "high",
			lowestLevel:   "low",
		}
	case strings.Contains(base, "gemini-3.1-pro"):
		return gemini3Capability{
			allowedLevels: []string{"low", "medium", "high"},
			defaultLevel:  "high",
			lowestLevel:   "low",
		}
	default:
		return gemini3Capability{
			allowedLevels: []string{"low", "medium", "high"},
			defaultLevel:  "high",
			lowestLevel:   "low",
		}
	}
}

// resolveGemini3ThinkingLevel maps arbitrary input effort/level to a valid Gemini 3 thinking level based on model capabilities.
// Supported levels for current Gemini 3 models (3.7-flash, 3.1-pro, etc.): "low", "medium", "high".
// Claude's "xhigh" / "max" are normalized to "high".
// If "minimal" is requested on models that do not support it, it safely downgrades to lowest supported level ("low").
func resolveGemini3ThinkingLevel(modelName string, effort string) string {
	clean := strings.ToLower(strings.TrimSpace(effort))
	cap := getGemini3Capability(modelName)

	switch clean {
	case "xhigh", "max", "high":
		return "high"
	case "medium":
		return "medium"
	case "low":
		return "low"
	case "minimal":
		for _, allowed := range cap.allowedLevels {
			if allowed == "minimal" {
				return "minimal"
			}
		}
		return cap.lowestLevel
	case "none", "off":
		return cap.lowestLevel
	default:
		return cap.defaultLevel
	}
}

// budgetToGemini3ThinkingLevel converts numeric thinking budget to Gemini 3 thinking level (aligned with CCS: >=8192 => high, else medium).
func budgetToGemini3ThinkingLevel(modelName string, budgetTokens int) string {
	if budgetTokens >= 8192 {
		return resolveGemini3ThinkingLevel(modelName, "high")
	}
	return resolveGemini3ThinkingLevel(modelName, "medium")
}

// resolveGemini3NoThinkingConfig determines the Gemini 3 config for disabled/none/off/-nothinking requests.
// Sets thinkingLevel to the lowest supported level (e.g. "low") and includeThoughts to false.
func resolveGemini3NoThinkingConfig(modelName string) *dto.GeminiThinkingConfig {
	cap := getGemini3Capability(modelName)
	return &dto.GeminiThinkingConfig{
		IncludeThoughts: false,
		ThinkingLevel:   cap.lowestLevel,
	}
}

func isNew25ProModel(modelName string) bool {
	base := cleanModelName(modelName)
	return strings.HasPrefix(base, "gemini-2.5-pro") &&
		!strings.HasPrefix(base, "gemini-2.5-pro-preview-05-06") &&
		!strings.HasPrefix(base, "gemini-2.5-pro-preview-03-25")
}

func is25FlashLiteModel(modelName string) bool {
	base := cleanModelName(modelName)
	return strings.HasPrefix(base, "gemini-2.5-flash-lite")
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
