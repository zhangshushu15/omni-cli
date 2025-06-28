/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAIProvider } from './openaiProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import { ProviderAPIError } from './baseProvider.js';

/**
 * vLLM provider that extends OpenAI provider for compatibility
 * vLLM provides an OpenAI-compatible API, so we can inherit most functionality
 * and just override the configuration and capabilities
 */
export class VLLMProvider extends OpenAIProvider {
  // Note: inherits provider from OpenAIProvider but overrides behavior

  constructor(config: ProviderConfig) {
    // Handle baseURL - ensure it ends with /v1 for OpenAI compatibility
    let baseURL = config.baseURL || 'http://localhost:8000';
    if (!baseURL.endsWith('/v1')) {
      baseURL = baseURL.replace(/\/$/, '') + '/v1';
    }

    super({
      ...config,
      provider: LLMProvider.VLLM,
      baseURL,
      apiKey: config.apiKey || '', // vLLM can optionally use API keys for auth
    });
  }

  async initialize(): Promise<void> {
    // Override initialization to check vLLM health instead of just API key
    try {
      const isHealthy = await this.checkVLLMHealth();
      if (!isHealthy) {
        throw new Error(
          'vLLM server not running. Please start vLLM server first: python -m vllm.entrypoints.openai_api_server --model <model_name>',
        );
      }
    } catch (error) {
      throw new Error(
        `Failed to initialize vLLM provider: ${error instanceof Error ? error.message : 'Unknown error'}`,
      );
    }
  }

  supportsEmbeddings(): boolean {
    return false; // vLLM is primarily for text generation, not embeddings
  }

  protected requiresApiKey(): boolean {
    return false; // vLLM can run without API keys (depends on server configuration)
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

    // Add authorization header only if API key is provided
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

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
    return this.checkVLLMHealth();
  }

  // Override error handling to report correct provider name
  protected createAPIError(message: string, originalError?: Error): Error {
    return new ProviderAPIError(LLMProvider.VLLM, message, originalError);
  }

  private async checkVLLMHealth(): Promise<boolean> {
    try {
      // Check if vLLM server is running by calling the models endpoint
      const response = await this.makeOpenAIRequest('/models', null, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch (error) {
      console.warn(
        `vLLM health check failed for ${this.baseURL}: ${error instanceof Error ? error.message : 'Unknown error'}`,
      );
      return false;
    }
  }

  /**
   * Get available models from the vLLM server
   * This can be useful for dynamic model discovery
   */
  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await this.makeOpenAIRequest('/models', null, {
        method: 'GET',
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      return data.data?.map((model: { id: string }) => model.id) || [];
    } catch (error) {
      throw new ProviderAPIError(
        LLMProvider.VLLM,
        `Failed to get available models: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  /**
   * Get server information from vLLM
   * This can provide useful debugging information
   */
  async getServerInfo(): Promise<Record<string, unknown>> {
    try {
      // vLLM doesn't have a standard info endpoint, but we can get model info
      const models = await this.getAvailableModels();
      return {
        available_models: models,
        server_type: 'vLLM',
        base_url: this.baseURL,
      };
    } catch (error) {
      throw new ProviderAPIError(
        LLMProvider.VLLM,
        `Failed to get server info: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }
}
