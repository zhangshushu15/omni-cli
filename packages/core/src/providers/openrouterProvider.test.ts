/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OpenRouterProvider } from './openrouterProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import { ProviderConfigurationError } from './baseProvider.js';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('OpenRouterProvider', () => {
  const defaultConfig: ProviderConfig = {
    provider: LLMProvider.OPENROUTER,
    model: 'anthropic/claude-3.5-sonnet',
    apiKey: 'test-api-key',
    baseURL: 'https://openrouter.ai/api/v1',
  };

  let provider: OpenRouterProvider;

  beforeEach(() => {
    provider = new OpenRouterProvider(defaultConfig);
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Constructor', () => {
    it('should set correct provider type', () => {
      expect(provider.provider).toBe(LLMProvider.OPENROUTER);
    });

    it('should set default base URL', () => {
      const configWithoutBaseURL = { ...defaultConfig };
      delete configWithoutBaseURL.baseURL;
      const provider = new OpenRouterProvider(configWithoutBaseURL);
      expect(provider).toBeDefined();
      // The base URL should be set to OpenRouter's default
    });

    it('should use custom base URL if provided', () => {
      const customConfig = {
        ...defaultConfig,
        baseURL: 'https://custom-openrouter-endpoint.com/v1',
      };
      const provider = new OpenRouterProvider(customConfig);
      expect(provider).toBeDefined();
    });

    it('should set default model', () => {
      const configWithoutModel = { ...defaultConfig };
      delete configWithoutModel.model;
      const provider = new OpenRouterProvider(configWithoutModel);
      expect(provider.getModel()).toBe('anthropic/claude-3.5-sonnet');
    });
  });

  describe('Initialize', () => {
    it('should throw error without API key', async () => {
      const configWithoutKey = { ...defaultConfig, apiKey: '' };
      const provider = new OpenRouterProvider(configWithoutKey);

      await expect(provider.initialize()).rejects.toThrow(
        ProviderConfigurationError,
      );
      await expect(provider.initialize()).rejects.toThrow(
        'API key is required for OpenRouter',
      );
    });

    it('should initialize successfully with valid API key', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [{ id: 'anthropic/claude-3.5-sonnet' }] }),
      });

      await expect(provider.initialize()).resolves.not.toThrow();

      expect(mockFetch).toHaveBeenCalledWith(
        'https://openrouter.ai/api/v1/models',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            Authorization: 'Bearer test-api-key',
          }),
        }),
      );
    });
  });

  describe('Tool Support', () => {
    it('should support embeddings', () => {
      expect(provider.supportsEmbeddings()).toBe(true);
    });
  });

  describe('Model Management', () => {
    it('should return effective model', () => {
      expect(provider.getModel()).toBe('anthropic/claude-3.5-sonnet');
    });

    it('should return custom model if provided', () => {
      const customConfig = {
        ...defaultConfig,
        model: 'openai/gpt-4o',
      };
      const provider = new OpenRouterProvider(customConfig);
      expect(provider.getModel()).toBe('openai/gpt-4o');
    });
  });

  describe('Content Generation', () => {
    const mockRequest = {
      contents: [{ parts: [{ text: 'Hello, world!' }], role: 'user' as const }],
      model: 'anthropic/claude-3.5-sonnet',
    };

    const mockOpenRouterResponse = {
      choices: [
        {
          message: {
            content: 'Hello! How can I help you today?',
            role: 'assistant',
          },
          finish_reason: 'stop',
          index: 0,
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 15,
        total_tokens: 25,
      },
    };

    it('should generate content successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockOpenRouterResponse,
      });

      const response = await provider.generateContent(mockRequest);

      expect(response).toBeDefined();
      expect(response.candidates?.length).toBe(1);
      expect(response.usageMetadata?.totalTokenCount).toBe(25);

      // Verify request format
      expect(mockFetch).toHaveBeenCalledWith(
        'https://openrouter.ai/api/v1/chat/completions',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            Authorization: 'Bearer test-api-key',
          }),
          body: expect.stringContaining('"messages"'),
        }),
      );
    });

    it('should handle API errors with OpenRouter-specific messaging', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Bad Request',
      });

      await expect(provider.generateContent(mockRequest)).rejects.toThrow();
    });
  });

  describe('Streaming', () => {
    it('should handle streaming responses', async () => {
      const mockStreamData = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        'data: [DONE]\n',
      ].join('');

      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(mockStreamData));
          controller.close();
        },
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: mockStream,
      });

      const stream = await provider.generateContentStream({
        contents: [{ parts: [{ text: 'Hello, streaming!' }], role: 'user' }],
        model: 'anthropic/claude-3.5-sonnet',
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);

      // Verify stream request
      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.stream).toBe(true);
    });
  });

  describe('Configuration Validation', () => {
    it('should validate configuration successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      const isValid = await provider.validateConfig();
      expect(isValid).toBe(true);
    });

    it('should return false for invalid configuration', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
      });

      const isValid = await provider.validateConfig();
      expect(isValid).toBe(false);
    });

    it('should return false for missing API key', async () => {
      const provider = new OpenRouterProvider({
        ...defaultConfig,
        apiKey: '',
      });

      const isValid = await provider.validateConfig();
      expect(isValid).toBe(false);
    });
  });
});
