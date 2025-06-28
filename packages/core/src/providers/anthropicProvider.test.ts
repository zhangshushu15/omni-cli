/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AnthropicProvider } from './anthropicProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import {
  ProviderConfigurationError,
  ProviderAPIError,
} from './baseProvider.js';
import { FinishReason } from '@google/genai';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('AnthropicProvider', () => {
  const defaultConfig: ProviderConfig = {
    provider: LLMProvider.ANTHROPIC,
    model: 'claude-3-sonnet-20240229',
    apiKey: 'test-api-key',
    baseURL: 'https://api.anthropic.com',
    // apiVersion: '2023-06-01',
  };

  let provider: AnthropicProvider;

  beforeEach(() => {
    provider = new AnthropicProvider(defaultConfig);
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Constructor', () => {
    it('should create provider with default config', () => {
      expect(provider.provider).toBe(LLMProvider.ANTHROPIC);
      expect(provider.getModel()).toBe('claude-3-sonnet-20240229');
    });

    it('should use default baseURL if not provided', () => {
      const config = { ...defaultConfig };
      delete config.baseURL;
      const testProvider = new AnthropicProvider(config);
      expect(testProvider).toBeDefined();
    });

    it('should use default API version if not provided', () => {
      const config = { ...defaultConfig };
      // delete config.apiVersion;
      const testProvider = new AnthropicProvider(config);
      expect(testProvider).toBeDefined();
    });
  });

  describe('Initialization', () => {
    it('should fail to initialize without API key', async () => {
      const configWithoutKey = { ...defaultConfig, apiKey: '' };
      const testProvider = new AnthropicProvider(configWithoutKey);

      await expect(testProvider.initialize()).rejects.toThrow(
        ProviderConfigurationError,
      );
      await expect(testProvider.initialize()).rejects.toThrow(
        'API key is required',
      );
    });

    it('should initialize successfully with valid API key', async () => {
      await expect(provider.initialize()).resolves.not.toThrow();
    });
  });

  describe('Capabilities', () => {
    it('should report correct capabilities', () => {
      expect(provider.supportsEmbeddings()).toBe(false);
      expect(provider.supportsStreaming()).toBe(true);
      expect(provider.supportsTools()).toBe(true);
    });
  });

  describe('generateContent', () => {
    const mockRequest = {
      contents: [{ parts: [{ text: 'Hello, world!' }], role: 'user' as const }],
      model: 'claude-3-sonnet-20240229',
    };

    const mockAnthropicResponse = {
      id: 'msg_123',
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'Hello! How can I help you today?' }],
      model: 'claude-3-sonnet-20240229',
      stop_reason: 'end_turn',
      usage: {
        input_tokens: 10,
        output_tokens: 15,
      },
    };

    it('should generate content successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockAnthropicResponse,
      });

      const response = await provider.generateContent(mockRequest);

      expect(response).toBeDefined();
      expect(response.candidates).toBeDefined();
      expect(response.candidates!.length).toBe(1);
      const candidate = response.candidates![0];
      expect(candidate.content?.parts?.[0].text).toBe(
        'Hello! How can I help you today?',
      );
      expect(response.usageMetadata?.totalTokenCount).toBe(25);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.anthropic.com/v1/messages',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'X-Api-Key': 'test-api-key',
            'anthropic-version': '2023-06-01',
          }),
          body: expect.stringContaining('"messages"'),
        }),
      );
    });

    it('should handle API errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Bad Request',
      });

      await expect(provider.generateContent(mockRequest)).rejects.toThrow(
        ProviderAPIError,
      );
    });

    it('should convert messages correctly', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockAnthropicResponse,
      });

      const requestWithMultipleMessages = {
        contents: [
          { parts: [{ text: 'System message' }], role: 'system' as const },
          { parts: [{ text: 'User message' }], role: 'user' as const },
        ],
        model: 'claude-3-sonnet-20240229',
      };

      await provider.generateContent(requestWithMultipleMessages);

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.messages).toHaveLength(2);
      expect(requestBody.messages[0].role).toBe('system');
      expect(requestBody.messages[1].role).toBe('user');
    });
  });

  describe('generateContentStream', () => {
    it('should handle streaming response', async () => {
      const mockStreamData = [
        'data: {"type":"content_block_delta","delta":{"text":"Hello"},"index":0}\n',
        'data: {"type":"content_block_delta","delta":{"text":" there"},"index":0}\n',
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
        model: 'claude-3-sonnet-20240229',
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBe(2);
      const firstChunk = chunks[0];
      const secondChunk = chunks[1];
      expect(firstChunk.candidates).toBeDefined();
      expect(secondChunk.candidates).toBeDefined();
      expect(firstChunk.candidates![0].content?.parts?.[0].text).toBe('Hello');
      expect(secondChunk.candidates![0].content?.parts?.[0].text).toBe(
        ' there',
      );

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.stream).toBe(true);
    });
  });

  describe('countTokens', () => {
    it('should return estimated token count', async () => {
      const request = {
        contents: [
          { parts: [{ text: 'Test message' }], role: 'user' as const },
        ],
        model: 'claude-3-sonnet-20240229',
      };

      const result = await provider.countTokens(request);
      expect(result).toBeDefined();
      expect(result.totalTokens).toBeGreaterThan(0);
      expect(typeof result.totalTokens).toBe('number');
    });
  });

  describe('embedContent', () => {
    it('should throw error as embeddings are not supported', async () => {
      const request = {
        contents: [{ parts: [{ text: 'Test text' }], role: 'user' as const }],
        model: 'claude-3-sonnet-20240229',
      };

      await expect(provider.embedContent(request)).rejects.toThrow(
        ProviderAPIError,
      );
      await expect(provider.embedContent(request)).rejects.toThrow(
        'Embeddings not supported',
      );
    });
  });

  describe('Message conversion', () => {
    it('should handle string content', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [{ type: 'text', text: 'response' }],
          model: 'claude-3-sonnet-20240229',
          stop_reason: 'end_turn',
          usage: { input_tokens: 5, output_tokens: 5 },
        }),
      });

      await provider.generateContent({
        contents: 'Simple string message',
        model: 'claude-3-sonnet-20240229',
      });

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.messages).toHaveLength(1);
      expect(requestBody.messages[0].content).toBe('Simple string message');
      expect(requestBody.messages[0].role).toBe('user');
    });
  });

  describe('Finish reason conversion', () => {
    it('should convert stop_sequence to STOP', () => {
      const response = {
        id: 'msg_123',
        type: 'message' as const,
        role: 'assistant' as const,
        content: [{ type: 'text' as const, text: 'response' }],
        model: 'claude-3-sonnet-20240229',
        stop_reason: 'stop_sequence' as const,
        usage: { input_tokens: 5, output_tokens: 5 },
      };

      const result = provider['convertAnthropicToGeminiResponse'](response);
      expect(result.candidates?.[0].finishReason).toBe(FinishReason.STOP);
    });

    it('should convert max_tokens to MAX_TOKENS', () => {
      const response = {
        id: 'msg_123',
        type: 'message' as const,
        role: 'assistant' as const,
        content: [{ type: 'text' as const, text: 'response' }],
        model: 'claude-3-sonnet-20240229',
        stop_reason: 'max_tokens' as const,
        usage: { input_tokens: 5, output_tokens: 5 },
      };

      const result = provider['convertAnthropicToGeminiResponse'](response);
      expect(result.candidates?.[0].finishReason).toBe(FinishReason.MAX_TOKENS);
    });

    it('should convert end_turn to STOP', () => {
      const response = {
        id: 'msg_123',
        type: 'message' as const,
        role: 'assistant' as const,
        content: [{ type: 'text' as const, text: 'response' }],
        model: 'claude-3-sonnet-20240229',
        stop_reason: 'end_turn' as const,
        usage: { input_tokens: 5, output_tokens: 5 },
      };

      const result = provider['convertAnthropicToGeminiResponse'](response);
      expect(result.candidates?.[0].finishReason).toBe(FinishReason.STOP);
    });
  });

  describe('Configuration parameters', () => {
    it('should pass through generation config', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'msg_123',
          type: 'message',
          role: 'assistant',
          content: [{ type: 'text', text: 'response' }],
          model: 'claude-3-sonnet-20240229',
          stop_reason: 'end_turn',
          usage: { input_tokens: 5, output_tokens: 5 },
        }),
      });

      await provider.generateContent({
        contents: [{ parts: [{ text: 'test' }], role: 'user' }],
        model: 'claude-3-sonnet-20240229',
        config: {
          temperature: 0.8,
          maxOutputTokens: 1000,
          stopSequences: ['STOP'],
        },
      });

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.temperature).toBe(0.8);
      expect(requestBody.max_tokens).toBe(1000);
      expect(requestBody.stop_sequences).toEqual(['STOP']);
    });
  });
});
