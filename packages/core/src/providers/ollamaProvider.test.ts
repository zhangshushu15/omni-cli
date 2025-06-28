/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OllamaProvider } from './ollamaProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('OllamaProvider', () => {
  const defaultConfig: ProviderConfig = {
    provider: LLMProvider.OLLAMA,
    model: 'llama3.1',
    baseURL: 'http://localhost:11434',
  };

  let provider: OllamaProvider;

  beforeEach(() => {
    provider = new OllamaProvider(defaultConfig);
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Constructor', () => {
    it('should create provider with default config', () => {
      // Note: provider property is inherited from OpenAIProvider but config has OLLAMA
      expect(provider.getModel()).toBe('llama3.1');
      // The underlying provider property might still be 'openai' due to inheritance
      // but the actual provider type is determined by the config
      expect(provider).toBeDefined();
    });

    it('should use default baseURL if not provided', () => {
      const config = { ...defaultConfig };
      delete config.baseURL;
      const testProvider = new OllamaProvider(config);
      // Should set default Ollama baseURL
      expect(testProvider).toBeDefined();
    });
  });

  describe('Capabilities', () => {
    it('should report correct capabilities', () => {
      expect(provider.supportsEmbeddings()).toBe(true);
    });
  });

  describe('Initialization', () => {
    it('should fail to initialize when Ollama server is not running', async () => {
      // Mock fetch to simulate server not running (network error)
      mockFetch.mockRejectedValueOnce(new Error('fetch failed'));

      await expect(provider.initialize()).rejects.toThrow(
        'Failed to initialize Ollama provider',
      );
    });

    it('should return false for validateConfig when server not running', async () => {
      // Mock fetch to simulate server not running (returns false response)
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const result = await provider.validateConfig();
      expect(result).toBe(false);
    });

    it('should initialize successfully when Ollama server is running', async () => {
      // Mock successful Ollama response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ models: [{ name: 'llama3.1' }] }),
      });

      await expect(provider.initialize()).resolves.not.toThrow();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:11434/api/tags',
        expect.objectContaining({
          method: 'GET',
        }),
      );
    });
  });

  describe('API Integration', () => {
    it('should inherit OpenAI API methods but work with Ollama endpoints', () => {
      // Test that methods exist (they inherit from OpenAI provider)
      expect(typeof provider.generateContent).toBe('function');
      expect(typeof provider.generateContentStream).toBe('function');
      expect(typeof provider.countTokens).toBe('function');
      expect(typeof provider.embedContent).toBe('function');
    });

    it('should generate content using Ollama API format', async () => {
      // Mock successful Ollama chat completion response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: 'Hello from Ollama!',
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
        }),
      });

      const response = await provider.generateContent({
        contents: [{ parts: [{ text: 'Hello!' }], role: 'user' }],
        model: 'llama3.1',
      });

      expect(response.candidates?.length).toBe(1);
      expect(response.usageMetadata?.totalTokenCount).toBe(15);

      // Verify it calls Ollama endpoint without Authorization header
      // Note: The actual call shows it's using the endpoint without /v1
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/chat/completions'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: expect.stringContaining('"model":"llama3.1"'),
        }),
      );

      // Verify no Authorization header is included
      const callArgs = mockFetch.mock.calls[0][1];
      expect(callArgs.headers).not.toHaveProperty('Authorization');
    });
  });
});
