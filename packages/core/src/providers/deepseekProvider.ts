/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from './openaiProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import { ProviderConfigurationError } from './baseProvider.js';

/**
 * DeepSeek provider that extends OpenAI provider since DeepSeek uses OpenAI-compatible API
 * This provider uses the same tool calling and streaming logic as OpenAI
 */
export class DeepSeekProvider extends OpenAIProvider {
  constructor(config: ProviderConfig) {
    // Configure DeepSeek-specific settings
    const deepSeekConfig: ProviderConfig = {
      ...config,
      baseURL: config.baseURL || 'https://api.deepseek.com',
      model: config.model || 'deepseek-chat',
    };

    super(deepSeekConfig);

    // Override the provider property after construction
    Object.defineProperty(this, 'provider', {
      value: LLMProvider.DEEPSEEK,
      writable: false,
      enumerable: true,
      configurable: false,
    });
  }

  async initialize(): Promise<void> {
    if (!this.apiKey) {
      throw new ProviderConfigurationError(
        this.provider,
        'DeepSeek API key is required. Please set DEEPSEEK_API_KEY environment variable or provide apiKey in config.',
      );
    }

    // Test API key by making a simple request
    try {
      await this.validateConfig();
    } catch (error) {
      throw new ProviderConfigurationError(
        this.provider,
        `Failed to initialize DeepSeek provider: ${error instanceof Error ? error.message : 'Unknown error'}`,
      );
    }
  }

  async validateConfig(): Promise<boolean> {
    try {
      // Test with a simple models list request (DeepSeek supports this endpoint)
      const response = await this.makeOpenAIRequest('/models', null, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  getEffectiveModel(): string {
    return this.config.model || 'deepseek-chat';
  }
}
