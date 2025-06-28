/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
} from '@google/genai';
import { LLMProvider, ProviderConfig } from '../config/models.js';

/**
 * Base interface that all LLM providers must implement
 * This provides a consistent abstraction over different provider APIs
 */
export interface BaseProvider {
  // readonly provider: LLMProvider;
  readonly config: ProviderConfig;

  /**
   * Initialize the provider with configuration
   */
  initialize(): Promise<void>;

  /**
   * Generate content using the provider's API
   */
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  /**
   * Generate streaming content using the provider's API
   */
  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  /**
   * Count tokens for the given content
   */
  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  /**
   * Generate embeddings for the given content (if supported)
   */
  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  /**
   * Check if the provider supports embeddings
   */
  supportsEmbeddings(): boolean;

  /**
   * Validate configuration and connectivity
   */
  validateConfig(): Promise<boolean>;
}

/**
 * Abstract base class with common functionality
 */
export abstract class AbstractProvider implements BaseProvider {
  readonly config: ProviderConfig;

  constructor(config: ProviderConfig) {
    this.config = config;
  }

  abstract initialize(): Promise<void>;
  abstract generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;
  abstract generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;
  abstract countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse>;
  abstract embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse>;

  getProvider(): LLMProvider {
    return this.config.provider;
  }

  getModel(): string {
    return this.config.model || '';
  }

  supportsEmbeddings(): boolean {
    throw new Error('Not implemented');
  }

  async validateConfig(): Promise<boolean> {
    if (!this.config.apiKey && this.requiresApiKey()) {
      return false;
    }
    return true;
  }

  protected requiresApiKey(): boolean {
    return (
      this.getProvider() !== LLMProvider.OLLAMA &&
      this.getProvider() !== LLMProvider.VLLM
    );
  }
}

/**
 * Error types for provider operations
 */
export class ProviderError extends Error {
  readonly provider: LLMProvider;
  readonly originalError?: Error;

  constructor(message: string, provider: LLMProvider, originalError?: Error) {
    super(message);
    this.name = 'ProviderError';
    this.provider = provider;
    this.originalError = originalError;
  }
}

export class ProviderConfigurationError extends ProviderError {
  constructor(provider: LLMProvider, message: string) {
    super(`Configuration error for ${provider}: ${message}`, provider);
    this.name = 'ProviderConfigurationError';
  }
}

export class ProviderAPIError extends ProviderError {
  constructor(provider: LLMProvider, message: string, originalError?: Error) {
    super(`API error for ${provider}: ${message}`, provider, originalError);
    this.name = 'ProviderAPIError';
  }
}
