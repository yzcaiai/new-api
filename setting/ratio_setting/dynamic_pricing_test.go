package ratio_setting

import "testing"

func TestResolveTextPricingRatiosGPT54Threshold(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.4-2026-03-05",
		272000,
		TextPricingModeStandard,
		1.25,
		6,
		0.1,
	)
	if applied {
		t.Fatalf("expected short-context pricing at 272000 input tokens")
	}
	if modelRatio != 1.25 || completionRatio != 6 || cacheRatio != 0.1 {
		t.Fatalf("unexpected short pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}

	modelRatio, completionRatio, cacheRatio, applied = ResolveTextPricingRatios(
		"gpt-5.4-2026-03-05",
		272001,
		TextPricingModeStandard,
		1.25,
		6,
		0.1,
	)
	if !applied {
		t.Fatalf("expected long-context pricing above 272000 input tokens")
	}
	if modelRatio != 2.5 || completionRatio != 4.5 || cacheRatio != 0.1 {
		t.Fatalf("unexpected long pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}

func TestResolveTextPricingRatiosGPT55Threshold(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.5-2026-04-24",
		272000,
		TextPricingModeStandard,
		2.5,
		6,
		0.1,
	)
	if applied {
		t.Fatalf("expected short-context pricing at 272000 input tokens")
	}
	if modelRatio != 2.5 || completionRatio != 6 || cacheRatio != 0.1 {
		t.Fatalf("unexpected short pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}

	modelRatio, completionRatio, cacheRatio, applied = ResolveTextPricingRatios(
		"gpt-5.5-2026-04-24",
		272001,
		TextPricingModeStandard,
		2.5,
		6,
		0.1,
	)
	if !applied {
		t.Fatalf("expected long-context pricing above 272000 input tokens")
	}
	if modelRatio != 5 || completionRatio != 4.5 || cacheRatio != 0.1 {
		t.Fatalf("unexpected long pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}

func TestResolveTextPricingRatiosGPT55Pro(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.5-pro-2026-04-24",
		272001,
		TextPricingModeStandard,
		15,
		6,
		0.1,
	)
	if !applied {
		t.Fatalf("expected gpt-5.5-pro long-context pricing")
	}
	if modelRatio != 30 || completionRatio != 4.5 || cacheRatio != 0.1 {
		t.Fatalf("unexpected gpt-5.5-pro long pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}

func TestResolveTextPricingRatiosGPT54FlexShortAndLong(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.4",
		100,
		TextPricingModeFlex,
		1.25,
		6,
		0.1,
	)
	if applied {
		t.Fatalf("did not expect long-context pricing at short flex input size")
	}
	if modelRatio != 0.625 || completionRatio != 6 || cacheRatio != 0.1 {
		t.Fatalf("unexpected flex short pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}

	modelRatio, completionRatio, cacheRatio, applied = ResolveTextPricingRatios(
		"gpt-5.4",
		272001,
		TextPricingModeFlex,
		1.25,
		6,
		0.1,
	)
	if !applied {
		t.Fatalf("expected long-context pricing at flex long input size")
	}
	if modelRatio != 1.25 || completionRatio != 4.5 || cacheRatio != 0.1 {
		t.Fatalf("unexpected flex long pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}

func TestResolveTextPricingRatiosGPT54BatchLong(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.4",
		272001,
		TextPricingModeBatch,
		1.25,
		6,
		0.1,
	)
	if !applied {
		t.Fatalf("expected long-context pricing at batch long input size")
	}
	if modelRatio != 1.25 || completionRatio != 4.5 || cacheRatio != 0.1 {
		t.Fatalf("unexpected batch long pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}

func TestResolveTextPricingRatiosGPT54ProAndPriority(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.4-pro",
		272001,
		TextPricingModeStandard,
		15,
		6,
		1,
	)
	if !applied {
		t.Fatalf("expected gpt-5.4-pro long-context pricing")
	}
	if modelRatio != 30 || completionRatio != 4.5 || cacheRatio != 1 {
		t.Fatalf("unexpected gpt-5.4-pro long pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}

	modelRatio, completionRatio, cacheRatio, applied = ResolveTextPricingRatios(
		"gpt-5.4",
		272001,
		TextPricingModePriority,
		1.25,
		6,
		0.1,
	)
	if applied {
		t.Fatalf("priority should keep current pricing because no official long-context priority price is configured")
	}
	if modelRatio != 1.25 || completionRatio != 6 || cacheRatio != 0.1 {
		t.Fatalf("unexpected priority pricing fallback: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}

func TestResolveTextPricingRatiosPreservesConfiguredBaseRatios(t *testing.T) {
	modelRatio, completionRatio, cacheRatio, applied := ResolveTextPricingRatios(
		"gpt-5.4",
		272001,
		TextPricingModeStandard,
		3,
		8,
		0.2,
	)
	if !applied {
		t.Fatalf("expected long-context pricing for configured base ratios")
	}
	if modelRatio != 6 || completionRatio != 6 || cacheRatio != 0.2 {
		t.Fatalf("unexpected configured-ratio pricing: model=%v completion=%v cache=%v", modelRatio, completionRatio, cacheRatio)
	}
}
