# Omni CLI

![Omni CLI Screenshot](./docs/assets/omni-screenshot.png)

Omni CLI is a fork of [Gemini CLI](https://github.com/google-gemini/gemini-cli) that also works with many other LLM providers. Right now, in addition to Gemini, it supports OpenAI, Anthropic, DeepSeek, OpenRouter, Ollama, and vLLM.

The purpose of this work is to test the agentic coding capability of models other than Gemini. As of 7/7/25, there's a huge difference in quality between frontier models and other ones.

To install and run, use:

```bash
npm install -g @zhangshushu15/omni-cli

# An example. You need to set up your ollama model.
omni --provider ollama --base-url http://localhost:11434 --model qwen3:32b
```

To run with original Gemini models, simply do:

```bash
omni
```

To see other providers:

```bash
omni --list--providers
```

To set the API keys for OpenAI, Anthropic, DeepSeek, and OpenRouter, use the following environment variables:
```bash
export OPENAI_API_KEY=sk-xxx
export ANTHROPIC_API_KEY=sk-xxx
export DEEPSEEK_API_KEY=sk-xxx
export OPENROUTER_API_KEY=sk-xxx
```

Or add them to your $HOME/.env file.

Please refer to Gemini CLI's documention for other usage. All of Gemini CLI's functions should be the same.

Please file bugs or feature requests. Thanks!
