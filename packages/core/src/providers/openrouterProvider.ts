/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from './openaiProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import { ProviderConfigurationError } from './baseProvider.js';

/**
 * OpenRouter provider implementation extending OpenAI provider
 *
 * OpenRouter provides access to multiple AI models through an OpenAI-compatible API.
 * This provider leverages the existing OpenAI implementation while providing
 * OpenRouter-specific configuration and validation.
 */
export class OpenRouterProvider extends OpenAIProvider {
  constructor(config: ProviderConfig) {
    // Set OpenRouter-specific defaults
    const openrouterConfig: ProviderConfig = {
      baseURL: 'https://openrouter.ai/api/v1',
      model: 'anthropic/claude-3.5-sonnet', // Default to a popular model
      ...config,
    };

    super(openrouterConfig);

    // Override the provider property to be OpenRouter
    Object.defineProperty(this, 'provider', {
      value: LLMProvider.OPENROUTER,
      writable: false,
      enumerable: true,
      configurable: false,
    });
  }

  /**
   * Initialize the OpenRouter provider
   */
  async initialize(): Promise<void> {
    if (!this.config.apiKey) {
      throw new ProviderConfigurationError(
        LLMProvider.OPENROUTER,
        'API key is required for OpenRouter. Set OPENROUTER_API_KEY environment variable or provide apiKey in config.',
      );
    }

    // Use the parent's initialization which validates by listing models
    await super.initialize();
  }

  /**
   * Validate the OpenRouter configuration
   */
  async validateConfig(): Promise<boolean> {
    try {
      if (!this.config.apiKey) {
        return false;
      }

      // Use the parent's validation which tries to list models
      return await super.validateConfig();
    } catch {
      return false;
    }
  }

  /**
   * Override to provide OpenRouter-specific error messages
   */
  protected getProviderSpecificErrorMessage(error: unknown): string {
    return `OpenRouter API error: ${error}`;
  }
}
