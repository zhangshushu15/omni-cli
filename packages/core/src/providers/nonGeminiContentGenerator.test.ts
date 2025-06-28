/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NonGeminiContentGenerator } from './nonGeminiContentGenerator.js';
import { BaseProvider, ProviderConfigurationError } from './baseProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  FinishReason,
} from '@google/genai';

// Mock all provider imports - these must be defined before other code
vi.mock('./openaiProvider.js', () => ({
  OpenAIProvider: class implements BaseProvider {
    readonly provider: LLMProvider;

    constructor(public config: ProviderConfig) {
      this.provider = config.provider;
    }

    async initialize(): Promise<void> {
      if (!this.config.apiKey && this.config.provider !== LLMProvider.OLLAMA) {
        throw new ProviderConfigurationError(
          this.provider,
          'API key is required for testing',
        );
      }
    }

    async generateContent(
      _request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            parts: [{ text: 'Mock response' }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
        },
      ];
      return response;
    }

    async generateContentStream(
      _request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      async function* mockStream() {
        const response1 = new GenerateContentResponse();
        response1.candidates = [
          {
            content: {
              parts: [{ text: 'Mock' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response1;

        const response2 = new GenerateContentResponse();
        response2.candidates = [
          {
            content: {
              parts: [{ text: ' stream' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response2;
      }
      return mockStream();
    }

    async countTokens(
      _request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      return {
        totalTokens: 42,
      };
    }

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      return {
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      };
    }

    supportsEmbeddings(): boolean {
      return true;
    }

    supportsStreaming(): boolean {
      return true;
    }

    supportsTools(): boolean {
      return true;
    }

    getEffectiveModel(): string {
      return this.config.model || 'mock-model';
    }

    async validateConfig(): Promise<boolean> {
      return true;
    }
  },
}));

vi.mock('./anthropicProvider.js', () => ({
  AnthropicProvider: class implements BaseProvider {
    readonly provider: LLMProvider;

    constructor(public config: ProviderConfig) {
      this.provider = config.provider;
    }

    async initialize(): Promise<void> {
      if (!this.config.apiKey && this.config.provider !== LLMProvider.OLLAMA) {
        throw new ProviderConfigurationError(
          this.provider,
          'API key is required for testing',
        );
      }
    }

    async generateContent(
      _request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            parts: [{ text: 'Mock response' }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
        },
      ];
      return response;
    }

    async generateContentStream(
      _request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      async function* mockStream() {
        const response1 = new GenerateContentResponse();
        response1.candidates = [
          {
            content: {
              parts: [{ text: 'Mock' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response1;

        const response2 = new GenerateContentResponse();
        response2.candidates = [
          {
            content: {
              parts: [{ text: ' stream' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response2;
      }
      return mockStream();
    }

    async countTokens(
      _request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      return {
        totalTokens: 42,
      };
    }

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      return {
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      };
    }

    supportsEmbeddings(): boolean {
      return true;
    }

    supportsStreaming(): boolean {
      return true;
    }

    supportsTools(): boolean {
      return true;
    }

    getEffectiveModel(): string {
      return this.config.model || 'mock-model';
    }

    async validateConfig(): Promise<boolean> {
      return true;
    }
  },
}));

vi.mock('./deepseekProvider.js', () => ({
  DeepSeekProvider: class implements BaseProvider {
    readonly provider: LLMProvider;

    constructor(public config: ProviderConfig) {
      this.provider = config.provider;
    }

    async initialize(): Promise<void> {
      if (!this.config.apiKey && this.config.provider !== LLMProvider.OLLAMA) {
        throw new ProviderConfigurationError(
          this.provider,
          'API key is required for testing',
        );
      }
    }

    async generateContent(
      _request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            parts: [{ text: 'Mock response' }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
        },
      ];
      return response;
    }

    async generateContentStream(
      _request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      async function* mockStream() {
        const response1 = new GenerateContentResponse();
        response1.candidates = [
          {
            content: {
              parts: [{ text: 'Mock' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response1;

        const response2 = new GenerateContentResponse();
        response2.candidates = [
          {
            content: {
              parts: [{ text: ' stream' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response2;
      }
      return mockStream();
    }

    async countTokens(
      _request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      return {
        totalTokens: 42,
      };
    }

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      return {
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      };
    }

    supportsEmbeddings(): boolean {
      return true;
    }

    supportsStreaming(): boolean {
      return true;
    }

    supportsTools(): boolean {
      return true;
    }

    getEffectiveModel(): string {
      return this.config.model || 'mock-model';
    }

    async validateConfig(): Promise<boolean> {
      return true;
    }
  },
}));

vi.mock('./openrouterProvider.js', () => ({
  OpenRouterProvider: class implements BaseProvider {
    readonly provider: LLMProvider;

    constructor(public config: ProviderConfig) {
      this.provider = config.provider;
    }

    async initialize(): Promise<void> {
      if (!this.config.apiKey && this.config.provider !== LLMProvider.OLLAMA) {
        throw new ProviderConfigurationError(
          this.provider,
          'API key is required for testing',
        );
      }
    }

    async generateContent(
      _request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            parts: [{ text: 'Mock response' }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
        },
      ];
      return response;
    }

    async generateContentStream(
      _request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      async function* mockStream() {
        const response1 = new GenerateContentResponse();
        response1.candidates = [
          {
            content: {
              parts: [{ text: 'Mock' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response1;

        const response2 = new GenerateContentResponse();
        response2.candidates = [
          {
            content: {
              parts: [{ text: ' stream' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response2;
      }
      return mockStream();
    }

    async countTokens(
      _request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      return {
        totalTokens: 42,
      };
    }

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      return {
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      };
    }

    supportsEmbeddings(): boolean {
      return true;
    }

    supportsStreaming(): boolean {
      return true;
    }

    supportsTools(): boolean {
      return true;
    }

    getEffectiveModel(): string {
      return this.config.model || 'mock-model';
    }

    async validateConfig(): Promise<boolean> {
      return true;
    }
  },
}));

vi.mock('./ollamaProvider.js', () => ({
  OllamaProvider: class implements BaseProvider {
    readonly provider: LLMProvider;

    constructor(public config: ProviderConfig) {
      this.provider = config.provider;
    }

    async initialize(): Promise<void> {
      if (!this.config.apiKey && this.config.provider !== LLMProvider.OLLAMA) {
        throw new ProviderConfigurationError(
          this.provider,
          'API key is required for testing',
        );
      }
    }

    async generateContent(
      _request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            parts: [{ text: 'Mock response' }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
        },
      ];
      return response;
    }

    async generateContentStream(
      _request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      async function* mockStream() {
        const response1 = new GenerateContentResponse();
        response1.candidates = [
          {
            content: {
              parts: [{ text: 'Mock' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response1;

        const response2 = new GenerateContentResponse();
        response2.candidates = [
          {
            content: {
              parts: [{ text: ' stream' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response2;
      }
      return mockStream();
    }

    async countTokens(
      _request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      return {
        totalTokens: 42,
      };
    }

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      return {
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      };
    }

    supportsEmbeddings(): boolean {
      return true;
    }

    supportsStreaming(): boolean {
      return true;
    }

    supportsTools(): boolean {
      return true;
    }

    getEffectiveModel(): string {
      return this.config.model || 'mock-model';
    }

    async validateConfig(): Promise<boolean> {
      return true;
    }
  },
}));

vi.mock('./vllmProvider.js', () => ({
  VLLMProvider: class implements BaseProvider {
    readonly provider: LLMProvider;

    constructor(public config: ProviderConfig) {
      this.provider = config.provider;
    }

    async initialize(): Promise<void> {
      if (!this.config.apiKey && this.config.provider !== LLMProvider.OLLAMA) {
        throw new ProviderConfigurationError(
          this.provider,
          'API key is required for testing',
        );
      }
    }

    async generateContent(
      _request: GenerateContentParameters,
    ): Promise<GenerateContentResponse> {
      const response = new GenerateContentResponse();
      response.candidates = [
        {
          content: {
            parts: [{ text: 'Mock response' }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
        },
      ];
      return response;
    }

    async generateContentStream(
      _request: GenerateContentParameters,
    ): Promise<AsyncGenerator<GenerateContentResponse>> {
      async function* mockStream() {
        const response1 = new GenerateContentResponse();
        response1.candidates = [
          {
            content: {
              parts: [{ text: 'Mock' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response1;

        const response2 = new GenerateContentResponse();
        response2.candidates = [
          {
            content: {
              parts: [{ text: ' stream' }],
              role: 'model',
            },
            finishReason: FinishReason.STOP,
            index: 0,
          },
        ];
        yield response2;
      }
      return mockStream();
    }

    async countTokens(
      _request: CountTokensParameters,
    ): Promise<CountTokensResponse> {
      return {
        totalTokens: 42,
      };
    }

    async embedContent(
      _request: EmbedContentParameters,
    ): Promise<EmbedContentResponse> {
      return {
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      };
    }

    supportsEmbeddings(): boolean {
      return true;
    }

    supportsStreaming(): boolean {
      return true;
    }

    supportsTools(): boolean {
      return true;
    }

    getEffectiveModel(): string {
      return this.config.model || 'mock-model';
    }

    async validateConfig(): Promise<boolean> {
      return true;
    }
  },
}));

describe('NonGeminiContentGenerator', () => {
  let generator: NonGeminiContentGenerator;

  const baseConfig = {
    model: 'test-model',
    apiKey: 'test-api-key',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Constructor and Provider Creation', () => {
    it('should create OpenAI provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.OPENAI,
      };
      generator = new NonGeminiContentGenerator(config);

      expect(generator.getProvider()).toBeDefined();
      expect(generator.getProvider().config.provider).toBe(LLMProvider.OPENAI);
    });

    it('should create Anthropic provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.ANTHROPIC,
      };
      generator = new NonGeminiContentGenerator(config);

      expect(generator.getProvider()).toBeDefined();
      expect(generator.getProvider().config.provider).toBe(
        LLMProvider.ANTHROPIC,
      );
    });

    it('should create DeepSeek provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.DEEPSEEK,
      };
      generator = new NonGeminiContentGenerator(config);

      expect(generator.getProvider()).toBeDefined();
      expect(generator.getProvider().config.provider).toBe(
        LLMProvider.DEEPSEEK,
      );
    });

    it('should create OpenRouter provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.OPENROUTER,
      };
      generator = new NonGeminiContentGenerator(config);

      expect(generator.getProvider()).toBeDefined();
      expect(generator.getProvider().config.provider).toBe(
        LLMProvider.OPENROUTER,
      );
    });

    it('should create Ollama provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.OLLAMA,
      };
      generator = new NonGeminiContentGenerator(config);

      expect(generator.getProvider()).toBeDefined();
      expect(generator.getProvider().config.provider).toBe(LLMProvider.OLLAMA);
    });

    it('should create VLLM provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.VLLM,
      };
      generator = new NonGeminiContentGenerator(config);

      expect(generator.getProvider()).toBeDefined();
      expect(generator.getProvider().config.provider).toBe(LLMProvider.VLLM);
    });

    it('should throw error for Gemini provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.GEMINI,
      };

      expect(() => new NonGeminiContentGenerator(config)).toThrow(
        ProviderConfigurationError,
      );
      expect(() => new NonGeminiContentGenerator(config)).toThrow(
        'Gemini uses code assist and is not supported here.',
      );
    });

    it('should throw error for unknown provider', () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: 'unknown-provider' as LLMProvider,
      };

      expect(() => new NonGeminiContentGenerator(config)).toThrow(
        ProviderConfigurationError,
      );
      expect(() => new NonGeminiContentGenerator(config)).toThrow(
        'Unknown provider: unknown-provider',
      );
    });
  });

  describe('Provider Methods', () => {
    beforeEach(() => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.OPENAI,
      };
      generator = new NonGeminiContentGenerator(config);
    });

    describe('initialize', () => {
      it('should initialize provider successfully', async () => {
        await expect(generator.initialize()).resolves.not.toThrow();
      });

      it('should propagate provider initialization errors', async () => {
        const config: ProviderConfig = {
          provider: LLMProvider.OPENAI,
          model: 'test-model',
          apiKey: '', // Missing API key
        };
        const testGenerator = new NonGeminiContentGenerator(config);

        await expect(testGenerator.initialize()).rejects.toThrow(
          ProviderConfigurationError,
        );
      });
    });

    describe('getProvider', () => {
      it('should return the current provider', () => {
        const provider = generator.getProvider();
        expect(provider).toBeDefined();
        expect(provider.config.provider).toBe(LLMProvider.OPENAI);
      });
    });

    describe('generateContent', () => {
      it('should delegate to provider generateContent', async () => {
        const request: GenerateContentParameters = {
          contents: [{ parts: [{ text: 'Hello' }], role: 'user' }],
          model: 'test-model',
        };

        const response = await generator.generateContent(request);

        expect(response).toBeDefined();
        expect(response.candidates).toBeDefined();
        expect(response.candidates!.length).toBe(1);
        expect(response.candidates![0].content?.parts?.[0].text).toBe(
          'Mock response',
        );
      });
    });

    describe('generateContentStream', () => {
      it('should delegate to provider generateContentStream', async () => {
        const request: GenerateContentParameters = {
          contents: [{ parts: [{ text: 'Hello streaming' }], role: 'user' }],
          model: 'test-model',
        };

        const stream = await generator.generateContentStream(request);

        const chunks = [];
        for await (const chunk of stream) {
          chunks.push(chunk);
        }

        expect(chunks).toHaveLength(2);
        expect(chunks[0].candidates![0].content?.parts?.[0].text).toBe('Mock');
        expect(chunks[1].candidates![0].content?.parts?.[0].text).toBe(
          ' stream',
        );
      });
    });

    describe('countTokens', () => {
      it('should delegate to provider countTokens', async () => {
        const request: CountTokensParameters = {
          contents: [{ parts: [{ text: 'Hello' }], role: 'user' }],
          model: 'test-model',
        };

        const response = await generator.countTokens(request);

        expect(response).toBeDefined();
        expect(response.totalTokens).toBe(42);
      });
    });

    describe('embedContent', () => {
      it('should delegate to provider embedContent when supported', async () => {
        const request: EmbedContentParameters = {
          contents: [{ parts: [{ text: 'Hello' }] }],
          model: 'test-embedding-model',
        };

        const response = await generator.embedContent(request);

        expect(response).toBeDefined();
        expect(response.embeddings).toBeDefined();
        expect(response.embeddings![0].values).toEqual([0.1, 0.2, 0.3]);
      });

      it('should throw error when provider does not support embeddings', async () => {
        // Create a provider config that will use a mock that doesn't support embeddings
        const config: ProviderConfig = {
          ...baseConfig,
          provider: LLMProvider.OPENAI,
        };
        const testGenerator = new NonGeminiContentGenerator(config);

        // Mock the provider's supportsEmbeddings method to return false
        const mockProvider = testGenerator.getProvider();
        vi.spyOn(mockProvider, 'supportsEmbeddings').mockReturnValue(false);

        const request: EmbedContentParameters = {
          contents: [{ parts: [{ text: 'Hello' }] }],
          model: 'test-embedding-model',
        };

        await expect(testGenerator.embedContent(request)).rejects.toThrow(
          'Provider openai does not support embeddings',
        );
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle provider creation with minimal config', () => {
      const config: ProviderConfig = {
        provider: LLMProvider.OLLAMA, // Ollama doesn't require API key
      };

      expect(() => new NonGeminiContentGenerator(config)).not.toThrow();
    });

    it('should handle async operations with provider errors', async () => {
      const config: ProviderConfig = {
        ...baseConfig,
        provider: LLMProvider.OPENAI,
      };
      generator = new NonGeminiContentGenerator(config);

      // Mock provider to throw an error
      const mockProvider = generator.getProvider();
      vi.spyOn(mockProvider, 'generateContent').mockRejectedValue(
        new Error('Provider API error'),
      );

      const request: GenerateContentParameters = {
        contents: [{ parts: [{ text: 'Hello' }], role: 'user' }],
        model: 'test-model',
      };

      await expect(generator.generateContent(request)).rejects.toThrow(
        'Provider API error',
      );
    });
  });
});
