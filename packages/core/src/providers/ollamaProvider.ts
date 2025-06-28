/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from './openaiProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import { ProviderAPIError } from './baseProvider.js';

/**
 * Ollama provider that extends OpenAI provider for compatibility
 * Ollama provides an OpenAI-compatible API, so we can inherit most functionality
 * and just override the configuration and capabilities
 */
export class OllamaProvider extends OpenAIProvider {
  // Note: inherits provider from OpenAIProvider but overrides behavior

  constructor(config: ProviderConfig) {
    super({
      ...config,
      provider: LLMProvider.OLLAMA,
      baseURL: config.baseURL || 'http://localhost:11434/v1',
      apiKey: '', // Ollama doesn't require API keys
    });
  }

  async initialize(): Promise<void> {
    // Override initialization to check Ollama health instead of API key
    try {
      const isHealthy = await this.checkOllamaHealth();
      if (!isHealthy) {
        throw new Error(
          'Ollama server not running. Please start Ollama first: ollama serve',
        );
      }
    } catch (error) {
      throw new Error(
        `Failed to initialize Ollama provider: ${error instanceof Error ? error.message : 'Unknown error'}`,
      );
    }
  }

  protected requiresApiKey(): boolean {
    return false; // Ollama runs locally without API keys
  }

  protected async makeOpenAIRequest(
    endpoint: string,
    body: unknown,
    options: { method?: string; signal?: AbortSignal } = {},
  ): Promise<Response> {
    const { method = 'POST', signal } = options;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Don't add Authorization header for Ollama
    // Add any custom headers from config
    // if (this.config.customHeaders) {
    //   Object.assign(headers, this.config.customHeaders);
    // }

    const requestOptions: RequestInit = {
      method,
      headers,
      signal,
    };

    if (body && method !== 'GET') {
      requestOptions.body = JSON.stringify(body);
    }

    return fetch(`${this.baseURL}${endpoint}`, requestOptions);
  }

  async validateConfig(): Promise<boolean> {
    return this.checkOllamaHealth();
  }

  // Override error handling to report correct provider name
  protected createAPIError(message: string, originalError?: Error): Error {
    return new ProviderAPIError(LLMProvider.OLLAMA, message, originalError);
  }

  private async checkOllamaHealth(): Promise<boolean> {
    try {
      // Use Ollama's native API endpoint for health check
      // Convert OpenAI-compatible URL to native Ollama API URL
      const baseUrl = this.baseURL.replace(/\/v1$/, '');
      const response = await fetch(`${baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}
