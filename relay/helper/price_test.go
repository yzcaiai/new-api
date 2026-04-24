package helper

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/setting/ratio_setting"
	"github.com/QuantumNous/new-api/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func init() {
	ratio_setting.InitRatioSettings()
}

func TestModelPriceHelperPreConsumeUsesGPT54LongContextPricing(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.4-2026-03-05",
		RequestURLPath:  "/v1/chat/completions",
		Request:         &dto.GeneralOpenAIRequest{},
	}

	priceData, err := ModelPriceHelper(ctx, info, 272001, &types.TokenCountMeta{})
	require.NoError(t, err)
	require.False(t, priceData.UsePrice)
	require.Equal(t, 1.25, priceData.ModelRatio)
	require.Equal(t, 6.0, priceData.CompletionRatio)
	require.Equal(t, 0.1, priceData.CacheRatio)
	require.Equal(t, 680003, priceData.QuotaToPreConsume)
}

func TestModelPriceHelperPreConsumeUsesGPT55LongContextPricing(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.5-2026-04-24",
		RequestURLPath:  "/v1/chat/completions",
		Request:         &dto.GeneralOpenAIRequest{},
	}

	priceData, err := ModelPriceHelper(ctx, info, 272001, &types.TokenCountMeta{})
	require.NoError(t, err)
	require.False(t, priceData.UsePrice)
	require.Equal(t, 2.5, priceData.ModelRatio)
	require.Equal(t, 6.0, priceData.CompletionRatio)
	require.Equal(t, 0.1, priceData.CacheRatio)
	require.Equal(t, 1360005, priceData.QuotaToPreConsume)
}

func TestModelPriceHelperPreConsumeUsesFlexWhenServiceTierAllowed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.4",
		RequestURLPath:  "/v1/chat/completions",
		Request: &dto.GeneralOpenAIRequest{
			ServiceTier: json.RawMessage(`"flex"`),
		},
		ChannelMeta: &relaycommon.ChannelMeta{
			ChannelOtherSettings: dto.ChannelOtherSettings{
				AllowServiceTier: true,
			},
		},
	}

	priceData, err := ModelPriceHelper(ctx, info, 272001, &types.TokenCountMeta{})
	require.NoError(t, err)
	require.Equal(t, 340001, priceData.QuotaToPreConsume)
}

func TestModelPriceHelperPreConsumeUsesGPT54LongContextOutputRatio(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.4",
		RequestURLPath:  "/v1/chat/completions",
		Request:         &dto.GeneralOpenAIRequest{},
	}

	priceData, err := ModelPriceHelper(ctx, info, 272001, &types.TokenCountMeta{MaxTokens: 100})
	require.NoError(t, err)
	require.Equal(t, 681128, priceData.QuotaToPreConsume)
}

func TestModelPriceHelperPreConsumeIgnoresFlexWhenServiceTierFiltered(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.4",
		RequestURLPath:  "/v1/chat/completions",
		Request: &dto.GeneralOpenAIRequest{
			ServiceTier: json.RawMessage(`"flex"`),
		},
		ChannelMeta: &relaycommon.ChannelMeta{},
	}

	priceData, err := ModelPriceHelper(ctx, info, 272001, &types.TokenCountMeta{})
	require.NoError(t, err)
	require.Equal(t, 680003, priceData.QuotaToPreConsume)
}
