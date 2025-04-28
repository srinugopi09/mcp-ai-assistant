from typing import Optional, Dict, List, Any, Union
import asyncio
import time
import logging
import json

import streamlit as st
from mcp import ListToolsResult
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from custom_bedrock_llm import CustomBedrockLLM
from mcp_agent.human_input.types import (
    HumanInputRequest,
    HumanInputResponse,
)
from mcp_agent.config import MCPServerSettings
from functools import partial

# ───────────────────────── LOGGING — turn on deep MCP debug
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("mcp_agent").setLevel(logging.DEBUG)

# ───────────────────────── SESSION STATE BOOTSTRAP
if "mcp_server_names" not in st.session_state:
    # Built‑in server we always want
    # st.session_state["mcp_server_names"] = ["github"]
    st.session_state["mcp_server_names"] = [""]

if "_custom_servers" not in st.session_state:
    st.session_state["_custom_servers"] = {}

# ───────────────────────── HUMAN‑IN‑THE‑LOOP CALLBACK


async def streamlit_input_callback(request: HumanInputRequest) -> HumanInputResponse:
    approved = st.session_state.get("tool_approval", False)
    return HumanInputResponse(request_id=request.request_id,
                              response="Approved." if approved else "Denied.")

# Create the global MCP wrapper
app = MCPApp(name="ai_agent", human_input_callback=streamlit_input_callback)


# ───────────────────────── BEDROCK RESPONSE PARSER ───────────────────────────

def extract_text_from_bedrock_response(response: Any) -> str:
    """
    Extract text content from Bedrock response structure, handling various formats.

    Bedrock responses can have a complex nested structure:
    - Can be a dict with 'content' list
    - Content items can have 'text', 'toolUse', or 'toolResult' keys
    - Can be a string representation of a dict
    """
    logger = logging.getLogger("extract_text")
    logger.debug(f"Processing response type: {type(response)}")

    # Case 1: String response
    if isinstance(response, str):
        logger.debug("Processing string response")
        try:
            # Try to parse as JSON
            parsed = json.loads(response.replace("'", '"'))
            if isinstance(parsed, dict) and 'text' in parsed:
                return parsed['text']
        except (json.JSONDecodeError, ValueError):
            # If not valid JSON, just return the string
            pass
        return response

    # Case 2: Dict response (typical Bedrock message format)
    if isinstance(response, dict):
        logger.debug("Processing dict response")
        # Direct text field
        if 'text' in response:
            logger.debug("Found direct text field")
            return response['text']

        # Content list field
        if 'content' in response and isinstance(response['content'], list):
            logger.debug("Processing content list")
            text_parts = []
            for item in response['content']:
                if isinstance(item, dict):
                    if 'text' in item:
                        text_parts.append(item['text'])
                    elif 'toolUse' in item:
                        tool_info = item['toolUse']
                        text_parts.append(
                            f"[Tool: {tool_info.get('name', 'unknown')}]")
                    elif 'toolResult' in item:
                        result_info = item['toolResult']
                        result_content = result_info.get('content', '')
                        if isinstance(result_content, list):
                            for content_item in result_content:
                                if isinstance(content_item, dict) and 'text' in content_item:
                                    text_parts.append(content_item['text'])
                                else:
                                    text_parts.append(str(content_item))
                        else:
                            text_parts.append(str(result_content))
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            return ''.join(text_parts)

        # Role/content structure
        if 'role' in response and response.get('role') == 'assistant':
            logger.debug("Processing assistant role message")
            return extract_text_from_bedrock_response(response.get('content', ''))

    # Case 3: List of messages
    if isinstance(response, list):
        logger.debug("Processing list response")
        text_parts = []
        for item in response:
            text_parts.append(extract_text_from_bedrock_response(item))
        return ''.join(text_parts)

    # Fallback: convert to string
    logger.debug(f"Fallback: converting to string")
    return str(response)


# ───────────────────────── HELPERS

def format_list_tools_result(list_tools_result: ListToolsResult) -> str:
    """Pretty-print tool list, tolerating different MCP versions."""
    lines = []
    for t in list_tools_result.tools:
        server = getattr(t, "server_name", None)
        if server:
            lines.append(
                f"- **{t.name}**: {t.description} _(server: {server})")
        else:
            lines.append(f"- **{t.name}**: {t.description}")

    return " ".join(lines)


def get_bedrock_credentials_from_session():
    creds = {
        "aws_access_key_id": st.session_state.get("aws_access_key_id", ""),
        "aws_secret_access_key": st.session_state.get("aws_secret_access_key", ""),
        "aws_session_token": st.session_state.get("aws_session_token", ""),
        "region_name": st.session_state.get("aws_region", "us-east-1"),
        "model": st.session_state.get("aws_model", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
    }
    return {k: v for k, v in creds.items() if v}

# ───────────────────────── AGENT / LLM FACTORY


async def get_ai_agent():
    # 1. Agent -----------------------------------------------------------------
    if "agent" not in st.session_state:
        ai_agent = Agent(
            name="aiagent",
            instruction=(
                """You are a helpful AI assistant designed to provide accurate, thoughtful responses to user queries while leveraging external tools when appropriate. You have the ability to connect with remote Model Context Protocol (MCP) servers that extend your capabilities beyond your built-in knowledge.

                    ## Core Responsibilities

                    1. Understand user questions and respond with clear, helpful answers using your built-in knowledge.

                    2. Recognize when a question would benefit from accessing external data or executing functions through available MCP tools.

                    3. Make appropriate decisions about which MCP tools to use based on the user's needs.

                    4. Transparently communicate to users when you're using external tools to gather information.

                    5. Integrate information from tools seamlessly into your responses, providing cohesive and natural-sounding answers.

                    ## Using MCP Tools

                    When you encounter a query that requires information beyond your knowledge or capabilities:

                    1. Consider if any available MCP tools would help answer the query more effectively.

                    2. If a relevant tool exists, use it and clearly indicate to the user that you're fetching external information.

                    3. If multiple tools could help, prioritize the most relevant one or use them in a logical sequence.

                    4. If a tool returns an error or insufficient information, explain this to the user and offer alternative approaches.

                    5. Always verify that tool outputs are relevant to the user's query before incorporating them into your response.

                    ## Response Format

                    When using MCP tools, structure your responses like this:

                    1. Begin with a direct acknowledgment of the user's question.

                    2. Indicate when you're using an external tool: "Let me check that for you..." or "I'll use [tool name] to find this information..."

                    3. Present the information from the tool in a clear, organized way.

                    4. Add your own analysis, context, or explanation as needed to make the information more useful.

                    5. Conclude with any relevant follow-up suggestions or ask if the user needs additional information.

                    ## Privacy and Security

                    1. Only use MCP tools when necessary and relevant to the user's query.

                    2. Do not request or send sensitive personal information through MCP tools.

                    3. If a user requests information that would require sharing sensitive data with external tools, suggest alternative approaches.

                    4. Be transparent about the limitations of MCP tools and your knowledge.

                    ## Continuous Improvement

                    1. Learn from user interactions to better understand when tool usage is appropriate.

                    2. Adapt your responses based on user feedback about tool usage.

                    3. Maintain a helpful, friendly, and professional tone in all interactions.
            """
            ),
            server_names=st.session_state["mcp_server_names"],
            connection_persistence=False,
            human_input_callback=streamlit_input_callback,
        )
        await ai_agent.initialize()
        st.session_state["agent"] = ai_agent

    # 2. LLM -------------------------------------------------------------------
    model_option = st.session_state.get("model_choice", "Bedrock")
    should_reinit_llm = (
        "llm" not in st.session_state or
        st.session_state.get("last_model") != model_option or
        st.session_state.get("credentials_updated", False)
    )

    if should_reinit_llm:
        if model_option == "Anthropic":
            st.session_state["llm"] = await st.session_state["agent"].attach_llm(
                AnthropicAugmentedLLM
            )
        else:
            creds = get_bedrock_credentials_from_session()
            llm_factory = partial(CustomBedrockLLM, **creds)
            st.session_state["llm"] = await st.session_state["agent"].attach_llm(llm_factory)

        st.session_state["last_model"] = model_option
        st.session_state["credentials_updated"] = False

    return st.session_state["agent"], st.session_state["llm"]

# ───────────────────────── MAIN STREAMLIT COROUTINE


async def main():
    await app.initialize()

    # Re‑install custom servers after hot‑reloads
    for n, cfg in st.session_state["_custom_servers"].items():
        app.context.server_registry.registry[n] = cfg

    # ─── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.toggle("Approve Tool Usage", key="tool_approval", value=True)

        model_choice = st.selectbox(
            "Select LLM Model", ("Bedrock"), key="model_choice")

        if model_choice == "Bedrock":
            st.subheader("AWS Bedrock Configuration")
            with st.expander("AWS Credentials"):
                st.text_input("Access Key ID",
                              key="aws_access_key_id", type="password")
                st.text_input("Secret Access Key",
                              key="aws_secret_access_key", type="password")
                st.text_input("Session Token (optional)",
                              key="aws_session_token", type="password")
                st.text_input("AWS Region", key="aws_region",
                              value="us-east-1")
                st.text_input("Bedrock Model", key="aws_model",
                              value="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
                if st.button("Update Credentials"):
                    st.session_state["credentials_updated"] = True
                    st.session_state.pop("llm", None)
                    st.success("Credentials updated. Re‑initialising LLM…")
                    st.rerun()

        # Remote MCP server section -----------------------------------------
        st.markdown("---")
        st.subheader("Add remote MCP server")
        srv_name = st.text_input("Server name (unique key)")
        srv_url = st.text_input("Server URL (ends with /sse or /ws)")
        if st.button("Add Server"):
            if not srv_name or not srv_url:
                st.warning("Please fill in both fields.")
                st.stop()
            if srv_name in st.session_state["mcp_server_names"]:
                st.warning("Server name already exists.")
                st.stop()
            cfg = MCPServerSettings(
                name=srv_name, url=srv_url, transport="sse")
            st.session_state["mcp_server_names"].append(srv_name)
            st.session_state["_custom_servers"][srv_name] = cfg
            app.context.server_registry.registry[srv_name] = cfg
            # force Agent + tools to rebuild
            st.session_state.pop("agent", None)
            st.session_state.pop("llm", None)
            st.session_state.pop("tools_str", None)
            st.success(f"Added '{srv_name}'. Re‑initialising…")
            st.rerun()

    # ─── GET / CREATE AGENT + LLM ------------------------------------------
    agent, llm = await get_ai_agent()

    # ─── MESSAGE HISTORY + TOOL VIEW ---------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    # always recompute tools when Agent changes
    tools = await agent.list_tools()
    st.session_state["tools_str"] = format_list_tools_result(tools)

    with st.expander("View Tools"):
        st.markdown(st.session_state["tools_str"])

    for m in st.session_state["messages"]:
        st.chat_message(m["role"]).write(m["content"])

    # ─── CHAT INPUT ---------------------------------------------------------
    if prompt := st.chat_input("Type your message here…"):
        st.session_state["messages"].append(
            {"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.write("…thinking…")

            raw_response = await llm.generate(message=prompt)

            # Process the response using our new parser
            if isinstance(raw_response, str):
                collected = extract_text_from_bedrock_response(raw_response)
                placeholder.write(collected)
            else:
                # For streamed responses (list or generator)
                collected = ""
                for chunk in raw_response:
                    # Extract text from each chunk
                    chunk_text = extract_text_from_bedrock_response(chunk)
                    if chunk_text:
                        collected += chunk_text
                        placeholder.write(collected)

            st.session_state["messages"].append(
                {"role": "assistant", "content": collected})
            st.rerun()


# ───────────────────────── RUN ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
