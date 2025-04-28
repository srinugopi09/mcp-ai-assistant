import boto3
from boto3 import Session
from mcp_agent.workflows.llm.augmented_llm_bedrock import BedrockAugmentedLLM


class CustomBedrockLLM(BedrockAugmentedLLM):
    def __init__(
        self,
        *args,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
        region_name=None,
        model=None,
        **kwargs,
    ):
        # 1. Build a boto3.Session *first*
        session = Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name or "us-east-1",
        )

        self.bedrock_client = session.client("bedrock-runtime")

        # 2. Now let the parent finish configuring itself
        super().__init__(*args, **kwargs)

        # 3. Override the default model if the caller set one
        if model and self.default_request_params:
            self.default_request_params.model = model
