/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DeepSeekProvider } from './deepseekProvider.js';
import { LLMProvider } from '../config/models.js';
import { ProviderConfigurationError } from './baseProvider.js';

// Mock the fetch function
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('DeepSeekProvider', () => {
  let provider: DeepSeekProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    provider = new DeepSeekProvider({
      provider: LLMProvider.DEEPSEEK,
      apiKey: 'test-key',
      model: 'deepseek-chat',
    });
  });

  describe('Constructor', () => {
    it('should set correct provider type', () => {
      expect(provider.provider).toBe(LLMProvider.DEEPSEEK);
    });

    it('should set default base URL', () => {
      expect(provider.config.baseURL).toBe('https://api.deepseek.com');
    });

    it('should use custom base URL if provided', () => {
      const customProvider = new DeepSeekProvider({
        provider: LLMProvider.DEEPSEEK,
        apiKey: 'test-key',
        baseURL: 'https://custom.deepseek.com',
      });
      expect(customProvider.config.baseURL).toBe('https://custom.deepseek.com');
    });

    it('should set default model', () => {
      const providerWithoutModel = new DeepSeekProvider({
        provider: LLMProvider.DEEPSEEK,
        apiKey: 'test-key',
      });
      expect(providerWithoutModel.getEffectiveModel()).toBe('deepseek-chat');
    });
  });

  describe('Initialize', () => {
    it('should throw error without API key', async () => {
      const providerWithoutKey = new DeepSeekProvider({
        provider: LLMProvider.DEEPSEEK,
      });

      await expect(providerWithoutKey.initialize()).rejects.toThrow(
        ProviderConfigurationError,
      );
    });

    it('should initialize successfully with valid API key', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      await expect(provider.initialize()).resolves.not.toThrow();
    });
  });

  describe('Tool Support', () => {
    it('should support embeddings', () => {
      expect(provider.supportsEmbeddings()).toBe(true);
    });
  });

  describe('Model Management', () => {
    it('should return effective model', () => {
      expect(provider.getEffectiveModel()).toBe('deepseek-chat');
    });

    it('should return custom model if provided', () => {
      const customProvider = new DeepSeekProvider({
        provider: LLMProvider.DEEPSEEK,
        apiKey: 'test-key',
        model: 'deepseek-coder',
      });
      expect(customProvider.getEffectiveModel()).toBe('deepseek-coder');
    });
  });
});
