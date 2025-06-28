/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export const DEFAULT_GEMINI_MODEL = 'gemini-2.5-pro';
export const DEFAULT_GEMINI_FLASH_MODEL = 'gemini-2.5-flash';
export const DEFAULT_GEMINI_EMBEDDING_MODEL = 'gemini-embedding-001';

export enum LLMProvider {
  GEMINI = 'gemini',
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  OPENROUTER = 'openrouter',
  OLLAMA = 'ollama',
  VLLM = 'vllm',
  DEEPSEEK = 'deepseek',
}

export interface ProviderInfo {
  model: string;
  baseURL?: string;
}

export const PROVIDER_INFO: Record<LLMProvider, ProviderInfo> = {
  [LLMProvider.GEMINI]: {
    model: DEFAULT_GEMINI_MODEL,
  },
  [LLMProvider.OPENAI]: {
    model: 'gpt-4o',
  },
  [LLMProvider.ANTHROPIC]: {
    model: 'claude-4-sonnet-20250514',
  },
  [LLMProvider.DEEPSEEK]: {
    model: 'deepseek-reasoner',
  },
  [LLMProvider.OPENROUTER]: {
    model: 'anthropic/claude-sonnet-4',
  },
  [LLMProvider.OLLAMA]: {
    model: 'qwen3:0.6b',
    baseURL: 'http://localhost:11434/v1',
  },
  [LLMProvider.VLLM]: {
    model: 'Qwen/Qwen3-0.6B',
    baseURL: 'http://localhost:8000/v1',
  },
} as const;

export interface ProviderConfig {
  provider: LLMProvider;
  model?: string;
  apiKey?: string;
  baseURL?: string;
  apiVersion?: string;
}
