/**
 * @license
 * Copyright 2025 The Omni Code Project
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OpenAIProvider } from './openaiProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';
import { Type, ToolListUnion, Part } from '@google/genai';
import {
  ProviderConfigurationError,
  ProviderAPIError,
} from './baseProvider.js';

// Type for testing that exposes private methods
type OpenAIProviderForTesting = OpenAIProvider & {
  convertGeminiToolsToOpenAITools: (
    tools: ToolListUnion | undefined,
  ) => unknown;
  parseToolCallArguments: (args: string) => Record<string, unknown>;
  extractToolCallsFromParts: (parts: Part[]) => unknown;
  convertFinishReason: (reason: string) => unknown;
  convertGeminiToOpenAIMessages: (contents: unknown) => Array<{
    role: string;
    content?: string | null;
    tool_calls?: Array<{
      id: string;
      type: string;
      function: { name: string; arguments: string };
    }>;
    tool_call_id?: string;
  }>;
};

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('OpenAIProvider', () => {
  const defaultConfig: ProviderConfig = {
    provider: LLMProvider.OPENAI,
    model: 'gpt-4o',
    apiKey: 'test-api-key',
    baseURL: 'https://api.openai.com/v1',
  };

  let provider: OpenAIProvider;

  beforeEach(() => {
    provider = new OpenAIProvider(defaultConfig);
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Constructor', () => {
    it('should create provider with default config', () => {
      expect(provider.provider).toBe(LLMProvider.OPENAI);
      expect(provider.getModel()).toBe('gpt-4o');
    });

    it('should use default baseURL if not provided', () => {
      const config = { ...defaultConfig };
      delete config.baseURL;
      const provider = new OpenAIProvider(config);
      expect(provider).toBeDefined();
    });
  });

  describe('Initialization', () => {
    it('should fail to initialize without API key', async () => {
      const configWithoutKey = { ...defaultConfig, apiKey: '' };
      const provider = new OpenAIProvider(configWithoutKey);

      await expect(provider.initialize()).rejects.toThrow(
        ProviderConfigurationError,
      );
      await expect(provider.initialize()).rejects.toThrow(
        'API key is required',
      );
    });

    it('should initialize successfully with valid API key', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [{ id: 'gpt-4o' }] }),
      });

      await expect(provider.initialize()).resolves.not.toThrow();

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.openai.com/v1/models',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            Authorization: 'Bearer test-api-key',
          }),
        }),
      );
    });
  });

  describe('Capabilities', () => {
    it('should report correct capabilities', () => {
      expect(provider.supportsEmbeddings()).toBe(true);
    });
  });

  describe('generateContent', () => {
    const mockRequest = {
      contents: [{ parts: [{ text: 'Hello, world!' }], role: 'user' as const }],
      model: 'gpt-4o',
    };

    const mockOpenAIResponse = {
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
        json: async () => mockOpenAIResponse,
      });

      const response = await provider.generateContent(mockRequest);

      // Safe assertions without strict null checks
      expect(response).toBeDefined();
      expect(response.candidates?.length).toBe(1);
      expect(response.usageMetadata?.totalTokenCount).toBe(25);

      // Verify request format
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.openai.com/v1/chat/completions',
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
        json: async () => mockOpenAIResponse,
      });

      const requestWithMultipleMessages = {
        contents: [
          { parts: [{ text: 'System message' }], role: 'system' as const },
          { parts: [{ text: 'User message' }], role: 'user' as const },
        ],
        model: 'gpt-4o',
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
        model: 'gpt-4o',
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

  describe('countTokens', () => {
    it('should estimate tokens from character count', async () => {
      const request = {
        contents: [
          {
            parts: [{ text: 'This is a test message' }],
            role: 'user' as const,
          },
        ],
        model: 'gpt-4o',
      };

      const result = await provider.countTokens(request);

      expect(result.totalTokens).toBeGreaterThan(0);
      expect(result.totalTokens).toBeLessThan(100);
    });
  });

  describe('embedContent', () => {
    it('should generate embeddings successfully', async () => {
      const mockEmbeddingResponse = {
        data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockEmbeddingResponse,
      });

      const request = {
        contents: [{ parts: [{ text: 'Test text' }], role: 'user' as const }],
        model: 'text-embedding-3-small',
      };

      const response = await provider.embedContent(request);

      expect(response.embeddings?.length).toBe(1);

      // Verify request
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.openai.com/v1/embeddings',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"input"'),
        }),
      );
    });
  });

  describe('validateConfig', () => {
    it('should return true for valid configuration', async () => {
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
  });

  describe('Message conversion', () => {
    it('should handle string content', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            { message: { content: 'response' }, finish_reason: 'stop' },
          ],
        }),
      });

      await provider.generateContent({
        contents: 'Simple string message',
        model: 'gpt-4o',
      });

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.messages).toHaveLength(1);
      expect(requestBody.messages[0].content).toBe('Simple string message');
    });
  });

  describe('Configuration parameters', () => {
    it('should pass through generation config', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            { message: { content: 'response' }, finish_reason: 'stop' },
          ],
        }),
      });

      await provider.generateContent({
        contents: [{ parts: [{ text: 'test' }], role: 'user' }],
        model: 'gpt-4o',
        config: {
          temperature: 0.8,
          maxOutputTokens: 1000,
          stopSequences: ['STOP'],
        },
      });

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.temperature).toBe(0.8);
      expect(requestBody.max_tokens).toBe(1000);
      expect(requestBody.stop).toEqual(['STOP']);
    });
  });

  describe('Tool Support', () => {
    const mockTools = [
      {
        functionDeclarations: [
          {
            name: 'get_weather',
            description: 'Get weather information',
            parameters: {
              type: Type.OBJECT,
              properties: {
                location: { type: Type.STRING },
              },
              required: ['location'],
            },
          },
        ],
      },
    ];

    describe('Tool conversion', () => {
      it('should convert Gemini tools to OpenAI format', () => {
        const convertedTools = (
          provider as OpenAIProviderForTesting
        ).convertGeminiToolsToOpenAITools(mockTools);

        expect(convertedTools).toHaveLength(1);
        expect(
          Array.isArray(convertedTools) ? convertedTools[0] : undefined,
        ).toEqual({
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather information',
            parameters: {
              type: 'OBJECT',
              properties: {
                location: { type: 'STRING' },
              },
              required: ['location'],
            },
          },
        });
      });

      it('should return undefined for empty tools', () => {
        expect(
          (
            provider as OpenAIProviderForTesting
          ).convertGeminiToolsToOpenAITools([]),
        ).toBeUndefined();
        expect(
          (
            provider as OpenAIProviderForTesting
          ).convertGeminiToolsToOpenAITools(undefined),
        ).toBeUndefined();
      });

      it('should handle tools without function declarations', () => {
        const emptyTools = [{ functionDeclarations: [] }];
        expect(
          (
            provider as OpenAIProviderForTesting
          ).convertGeminiToolsToOpenAITools(emptyTools),
        ).toBeUndefined();
      });
    });

    describe('Tool call parsing', () => {
      it('should parse tool call arguments correctly', () => {
        const provider = new OpenAIProvider(defaultConfig);

        expect(
          (provider as OpenAIProviderForTesting).parseToolCallArguments(
            '{"location": "New York"}',
          ),
        ).toEqual({
          location: 'New York',
        });

        expect(
          (provider as OpenAIProviderForTesting).parseToolCallArguments(''),
        ).toEqual({});
        expect(
          (provider as OpenAIProviderForTesting).parseToolCallArguments(
            'invalid json',
          ),
        ).toEqual({});
      });

      it('should extract tool calls from parts', () => {
        const parts = [
          { text: 'Let me check the weather' },
          {
            functionCall: {
              id: 'call_123',
              name: 'get_weather',
              args: { location: 'New York' },
            },
          },
        ];

        const toolCalls = (
          provider as OpenAIProviderForTesting
        ).extractToolCallsFromParts(parts);

        expect(toolCalls).toHaveLength(1);
        expect(toolCalls[0]).toEqual({
          id: 'call_123',
          type: 'function',
          function: {
            name: 'get_weather',
            arguments: JSON.stringify({ location: 'New York' }),
          },
        });
      });

      it('should generate IDs for tool calls without them', () => {
        const parts = [
          {
            functionCall: {
              name: 'get_weather',
              args: { location: 'New York' },
            },
          },
        ];

        const toolCalls = (
          provider as OpenAIProviderForTesting
        ).extractToolCallsFromParts(parts);

        expect(toolCalls).toHaveLength(1);
        expect(toolCalls[0].id).toMatch(/^call_\d+_/);
        expect(toolCalls[0].function.name).toBe('get_weather');
      });
    });

    describe('generateContent with tools', () => {
      it('should include tools in request when provided', async () => {
        const mockResponse = {
          choices: [
            {
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [
                  {
                    id: 'call_123',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"location": "New York"}',
                    },
                  },
                ],
              },
              finish_reason: 'tool_calls',
              index: 0,
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 15, total_tokens: 25 },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        });

        const request = {
          contents: [
            {
              parts: [{ text: 'What is the weather in New York?' }],
              role: 'user' as const,
            },
          ],
          model: 'gpt-4o',
          config: { tools: mockTools },
        };

        const response = await provider.generateContent(request);

        // Verify tools were included in request
        const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(requestBody.tools).toBeDefined();
        expect(requestBody.tool_choice).toBe('auto');
        expect(requestBody.tools[0].function.name).toBe('get_weather');

        // Verify response contains function call
        expect(response.candidates?.[0].content?.parts?.[0]).toHaveProperty(
          'functionCall',
        );
        const functionCallPart = response.candidates?.[0].content
          ?.parts?.[0] as {
          functionCall: {
            id: string;
            name: string;
            args: Record<string, unknown>;
          };
        };
        expect(functionCallPart.functionCall.id).toBe('call_123');
        expect(functionCallPart.functionCall.name).toBe('get_weather');
        expect(functionCallPart.functionCall.args).toEqual({
          location: 'New York',
        });
      });

      it('should handle multiple tool calls in response', async () => {
        const mockResponse = {
          choices: [
            {
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [
                  {
                    id: 'call_123',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"location": "New York"}',
                    },
                  },
                  {
                    id: 'call_456',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"location": "London"}',
                    },
                  },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        });

        const response = await provider.generateContent({
          contents: [
            { parts: [{ text: 'Weather in NY and London?' }], role: 'user' },
          ],
          model: 'gpt-4o',
          config: { tools: mockTools },
        });

        expect(response.candidates?.[0].content?.parts).toHaveLength(2);
        const part0 = response.candidates?.[0].content?.parts?.[0] as {
          functionCall: { id: string };
        };
        const part1 = response.candidates?.[0].content?.parts?.[1] as {
          functionCall: { id: string };
        };
        expect(part0.functionCall.id).toBe('call_123');
        expect(part1.functionCall.id).toBe('call_456');
      });

      it('should handle mixed content and tool calls', async () => {
        const mockResponse = {
          choices: [
            {
              message: {
                role: 'assistant',
                content: 'Let me check the weather for you.',
                tool_calls: [
                  {
                    id: 'call_123',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"location": "New York"}',
                    },
                  },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        });

        const response = await provider.generateContent({
          contents: [
            { parts: [{ text: 'What is the weather?' }], role: 'user' },
          ],
          model: 'gpt-4o',
          config: { tools: mockTools },
        });

        expect(response.candidates?.[0].content?.parts).toHaveLength(2);
        const textPart = response.candidates?.[0].content?.parts?.[0] as {
          text: string;
        };
        const functionPart = response.candidates?.[0].content?.parts?.[1] as {
          functionCall: { name: string };
        };
        expect(textPart.text).toBe('Let me check the weather for you.');
        expect(functionPart.functionCall.name).toBe('get_weather');
      });
    });

    describe('Message conversion with tools', () => {
      it('should convert tool call messages correctly', () => {
        const contents = [
          {
            role: 'model' as const,
            parts: [
              {
                functionCall: {
                  id: 'call_123',
                  name: 'get_weather',
                  args: { location: 'New York' },
                },
              },
            ],
          },
        ];

        const messages = (
          provider as OpenAIProviderForTesting
        ).convertGeminiToOpenAIMessages(contents);

        expect(messages).toHaveLength(1);
        expect(messages[0]).toEqual({
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: 'call_123',
              type: 'function',
              function: {
                name: 'get_weather',
                arguments: JSON.stringify({ location: 'New York' }),
              },
            },
          ],
        });
      });

      it('should convert tool response messages correctly', () => {
        const contents = [
          {
            role: 'user' as const,
            parts: [
              {
                functionResponse: {
                  id: 'call_123',
                  name: 'get_weather',
                  response: { temperature: '72°F', condition: 'sunny' },
                },
              },
            ],
          },
        ];

        const messages = (
          provider as OpenAIProviderForTesting
        ).convertGeminiToOpenAIMessages(contents);

        expect(messages).toHaveLength(1);
        expect(messages[0]).toEqual({
          role: 'tool',
          content: JSON.stringify({ temperature: '72°F', condition: 'sunny' }),
          tool_call_id: 'call_123',
        });
      });

      it('should handle multiple tool responses', () => {
        const contents = [
          {
            role: 'user' as const,
            parts: [
              {
                functionResponse: {
                  id: 'call_123',
                  name: 'get_weather',
                  response: { temperature: '72°F' },
                },
              },
              {
                functionResponse: {
                  id: 'call_456',
                  name: 'get_weather',
                  response: { temperature: '15°C' },
                },
              },
            ],
          },
        ];

        const messages = (
          provider as OpenAIProviderForTesting
        ).convertGeminiToOpenAIMessages(contents);

        expect(messages).toHaveLength(2);
        expect(messages[0].role).toBe('tool');
        expect(messages[0].tool_call_id).toBe('call_123');
        expect(messages[1].role).toBe('tool');
        expect(messages[1].tool_call_id).toBe('call_456');
      });
    });

    describe('generateContentStream with tools', () => {
      it('should handle streaming tool calls', async () => {
        const mockStreamData = [
          'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}\n',
          'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather"}}]}}]}\n',
          'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"location\\":"}}]}}]}\n',
          'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"New York\\"}"}}]}}]}\n',
          'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n',
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
          contents: [
            { parts: [{ text: 'What is the weather?' }], role: 'user' },
          ],
          model: 'gpt-4o',
          config: { tools: mockTools },
        });

        const chunks = [];
        for await (const chunk of stream) {
          chunks.push(chunk);
        }

        // Should have received a chunk with the completed tool call
        expect(chunks.length).toBeGreaterThan(0);
        const toolCallChunk = chunks.find((chunk) =>
          chunk.candidates?.[0].content?.parts?.some(
            (part) => 'functionCall' in part,
          ),
        );
        expect(toolCallChunk).toBeDefined();

        const functionCallPart = toolCallChunk?.candidates?.[0].content
          ?.parts?.[0] as {
          functionCall: {
            id: string;
            name: string;
            args: Record<string, unknown>;
          };
        };
        expect(functionCallPart.functionCall.id).toBe('call_123');
        expect(functionCallPart.functionCall.name).toBe('get_weather');
        expect(functionCallPart.functionCall.args).toEqual({
          location: 'New York',
        });

        // Verify tools were included in request
        const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(requestBody.tools).toBeDefined();
        expect(requestBody.tool_choice).toBe('auto');
      });

      it('should complete tool call at end of stream', async () => {
        const mockStreamData = [
          'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NY\\"}"}}]}}]}\n',
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
          contents: [{ parts: [{ text: 'Weather?' }], role: 'user' }],
          model: 'gpt-4o',
          config: { tools: mockTools },
        });

        const chunks = [];
        for await (const chunk of stream) {
          chunks.push(chunk);
        }

        expect(chunks.length).toBeGreaterThan(0);
        const lastChunk = chunks[chunks.length - 1];
        const functionCallPart = lastChunk.candidates?.[0].content
          ?.parts?.[0] as {
          functionCall: { name: string; args: Record<string, unknown> };
        };
        expect(functionCallPart.functionCall.name).toBe('get_weather');
        expect(functionCallPart.functionCall.args).toEqual({ location: 'NY' });
      });
    });

    describe('Finish reason conversion', () => {
      it('should convert tool_calls finish reason', () => {
        const converted = (
          provider as OpenAIProviderForTesting
        ).convertFinishReason('tool_calls');
        expect(converted).toBeDefined();
      });

      it('should handle unknown finish reasons', () => {
        const converted = (
          provider as OpenAIProviderForTesting
        ).convertFinishReason('unknown_reason');
        expect(converted).toBeUndefined();
      });
    });
  });
});
