/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { VLLMProvider } from './vllmProvider.js';
import { LLMProvider } from '../config/models.js';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('VLLMProvider', () => {
  let provider: VLLMProvider;

  beforeEach(() => {
    mockFetch.mockClear();
    provider = new VLLMProvider({
      provider: LLMProvider.VLLM,
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      baseURL: 'http://localhost:8000/v1',
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create with default vLLM configuration', () => {
      expect(provider).toBeDefined();
      expect(provider.provider).toBe(LLMProvider.OPENAI); // Inherited from OpenAIProvider
    });

    it('should use custom baseURL when provided', () => {
      const customProvider = new VLLMProvider({
        provider: LLMProvider.VLLM,
        model: 'test-model',
        baseURL: 'http://custom-server:9000/v1',
      });
      expect(customProvider).toBeDefined();
    });

    it('should use default baseURL when not provided', () => {
      const defaultProvider = new VLLMProvider({
        provider: LLMProvider.VLLM,
        model: 'test-model',
      });
      expect(defaultProvider).toBeDefined();
    });
  });

  describe('initialize', () => {
    it('should initialize successfully when vLLM server is healthy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      await expect(provider.initialize()).resolves.toBeUndefined();
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/v1/models',
        expect.objectContaining({
          method: 'GET',
          signal: expect.any(AbortSignal),
        }),
      );
    });

    it('should throw error when vLLM server is not running', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Connection refused'));

      await expect(provider.initialize()).rejects.toThrow(
        'Failed to initialize vLLM provider',
      );
    });

    it('should throw error when vLLM server returns non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      await expect(provider.initialize()).rejects.toThrow(
        'vLLM server not running',
      );
    });
  });

  describe('capabilities', () => {
    it('should not support embeddings', () => {
      expect(provider.supportsEmbeddings()).toBe(false);
    });

    it('should not require API key by default', () => {
      expect(provider['requiresApiKey']()).toBe(false);
    });
  });

  describe('generateContent', () => {
    it('should generate content successfully', async () => {
      const mockResponse = {
        choices: [
          {
            message: {
              content: 'Hello, world!',
              role: 'assistant',
            },
            finish_reason: 'stop',
            index: 0,
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await provider.generateContent({
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      });

      expect(result.candidates?.[0]?.content?.parts?.[0]?.text).toBe(
        'Hello, world!',
      );
      expect(result.usageMetadata?.totalTokenCount).toBe(15);
    });

    it('should handle API errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Internal Server Error',
      });

      await expect(
        provider.generateContent({
          model: 'meta-llama/Llama-3.1-8B-Instruct',
          contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
        }),
      ).rejects.toThrow('Generation failed');
    });
  });

  describe('generateContentStream', () => {
    it('should generate streaming content', async () => {
      const mockStreamData = `data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n\n`;

      const mockResponse = {
        ok: true,
        body: {
          getReader: () => ({
            read: vi
              .fn()
              .mockResolvedValueOnce({
                done: false,
                value: new TextEncoder().encode(mockStreamData),
              })
              .mockResolvedValueOnce({
                done: true,
                value: undefined,
              }),
            releaseLock: vi.fn(),
          }),
        },
      };

      mockFetch.mockResolvedValueOnce(mockResponse);

      const stream = await provider.generateContentStream({
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks).toHaveLength(2);
      expect(chunks[0].candidates?.[0]?.content?.parts?.[0]?.text).toBe(
        'Hello',
      );
      expect(chunks[1].candidates?.[0]?.content?.parts?.[0]?.text).toBe(
        ' world',
      );
    });
  });

  describe('validateConfig', () => {
    it('should return true when server is healthy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      const isValid = await provider.validateConfig();
      expect(isValid).toBe(true);
    });

    it('should return false when server is not reachable', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Connection refused'));

      const isValid = await provider.validateConfig();
      expect(isValid).toBe(false);
    });
  });

  describe('getAvailableModels', () => {
    it('should return list of available models', async () => {
      const mockModelsResponse = {
        data: [
          { id: 'meta-llama/Llama-3.1-8B-Instruct', object: 'model' },
          { id: 'mistralai/Mistral-7B-Instruct-v0.3', object: 'model' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockModelsResponse,
      });

      const models = await provider.getAvailableModels();
      expect(models).toEqual([
        'meta-llama/Llama-3.1-8B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
      ]);
    });

    it('should handle empty model list', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      const models = await provider.getAvailableModels();
      expect(models).toEqual([]);
    });

    it('should throw error when models endpoint fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Server Error',
      });

      await expect(provider.getAvailableModels()).rejects.toThrow(
        'Failed to get available models',
      );
    });
  });

  describe('getServerInfo', () => {
    it('should return server information', async () => {
      const mockModelsResponse = {
        data: [{ id: 'meta-llama/Llama-3.1-8B-Instruct', object: 'model' }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockModelsResponse,
      });

      const info = await provider.getServerInfo();
      expect(info).toEqual({
        available_models: ['meta-llama/Llama-3.1-8B-Instruct'],
        server_type: 'vLLM',
        base_url: 'http://localhost:8000/v1',
      });
    });
  });

  describe('API key handling', () => {
    it('should work without API key', async () => {
      const providerWithoutKey = new VLLMProvider({
        provider: LLMProvider.VLLM,
        model: 'test-model',
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      await expect(providerWithoutKey.initialize()).resolves.toBeUndefined();
    });

    it('should include Authorization header when API key is provided', async () => {
      const providerWithKey = new VLLMProvider({
        provider: LLMProvider.VLLM,
        model: 'test-model',
        apiKey: 'test-api-key',
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ choices: [{ message: { content: 'test' } }] }),
      });

      await providerWithKey.generateContent({
        model: 'test-model',
        contents: [{ role: 'user', parts: [{ text: 'test' }] }],
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/chat/completions'),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-api-key',
          }),
        }),
      );
    });
  });

  describe('error handling', () => {
    it('should create proper vLLM API errors', () => {
      const error = provider['createAPIError']('Test error message');
      expect(error.message).toContain('Test error message');
    });
  });
});
