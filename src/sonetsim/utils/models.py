from camel.types import ModelPlatformType

MODEL_PLATFORM2BASE_URL: dict[ModelPlatformType, str] = {
    ModelPlatformType.OPENAI: "https://api.openai.com/v1",
    ModelPlatformType.ZHIPU: "https://open.bigmodel.cn/api/paas/v4",
}

MODEL_PLATFORM2API_KEY_ENV_NAME: dict[ModelPlatformType, str] = {
    ModelPlatformType.OPENAI: "OPENAI_API_KEY",
    ModelPlatformType.ZHIPU: "ZHIPUAI_API_KEY",
}
