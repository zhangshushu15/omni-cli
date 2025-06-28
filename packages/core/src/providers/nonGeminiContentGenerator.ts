/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseProvider, ProviderConfigurationError } from './baseProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
} from '@google/genai';
import { ContentGenerator } from '../core/contentGenerator.js';
import { OpenAIProvider } from './openaiProvider.js';
import { AnthropicProvider } from './anthropicProvider.js';
import { DeepSeekProvider } from './deepseekProvider.js';
import { OpenRouterProvider } from './openrouterProvider.js';
import { OllamaProvider } from './ollamaProvider.js';
import { VLLMProvider } from './vllmProvider.js';

/**
 * Central manager for all LLM providers
 * This class routes requests to the appropriate provider and provides a unified interface
 */
export class NonGeminiContentGenerator implements ContentGenerator {
  private provider: BaseProvider;

  constructor(config: ProviderConfig) {
    this.provider = this.createProvider(config);
  }

  /**
   * Set the provider for operations
   */
  async initialize(): Promise<void> {
    await this.provider.initialize();
  }

  /**
   * Get the current provider
   */
  getProvider(): BaseProvider {
    return this.provider;
  }

  /**
   * Generate content using the provider
   */
  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    return await this.getProvider().generateContent(request);
  }

  /**
   * Generate streaming content using the provider
   */
  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return await this.getProvider().generateContentStream(request);
  }

  /**
   * Count tokens using the provider
   */
  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    return await this.getProvider().countTokens(request);
  }

  /**
   * Generate embeddings using the provider
   */
  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const provider = this.getProvider();
    if (!provider.supportsEmbeddings()) {
      throw new Error(
        // TODO(Omni): provider.config.provider is weird?
        `Provider ${provider.config.provider} does not support embeddings`,
      );
    }
    return await provider.embedContent(request);
  }

  /**
   * Factory method to create provider instances
   */
  private createProvider(config: ProviderConfig): BaseProvider {
    switch (config.provider) {
      case LLMProvider.GEMINI:
        throw new ProviderConfigurationError(
          LLMProvider.GEMINI,
          'Gemini uses code assist and is not supported here.',
        );
      case LLMProvider.OPENAI:
        return new OpenAIProvider(config);
      case LLMProvider.ANTHROPIC:
        return new AnthropicProvider(config);
      case LLMProvider.DEEPSEEK:
        return new DeepSeekProvider(config);
      case LLMProvider.OPENROUTER:
        return new OpenRouterProvider(config);
      case LLMProvider.OLLAMA:
        return new OllamaProvider(config);
      case LLMProvider.VLLM:
        return new VLLMProvider(config);
      default:
        throw new ProviderConfigurationError(
          config.provider,
          `Unknown provider: ${config.provider}`,
        );
    }
  }
}
