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
  Content,
  ContentListUnion,
  FinishReason,
  Part,
  ToolListUnion,
  FunctionDeclaration,
} from '@google/genai';
import {
  AbstractProvider,
  ProviderAPIError,
  ProviderConfigurationError,
} from './baseProvider.js';
import { LLMProvider, ProviderConfig } from '../config/models.js';

// OpenAI API types
interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string; // Required for tool role messages
}

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface OpenAIChatRequest {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string[];
  frequency_penalty?: number;
  presence_penalty?: number;
  top_p?: number;
  tools?: OpenAITool[];
  tool_choice?:
    | 'none'
    | 'auto'
    | { type: 'function'; function: { name: string } };
}

interface OpenAIChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string | null;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  system_fingerprint?: string;
}

interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIEmbeddingRequest {
  model: string;
  input: string | string[];
}

interface OpenAIEmbeddingResponse {
  data: Array<{
    embedding: number[];
    index: number;
  }>;
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

/**
 * OpenAI provider that adapts OpenAI API to the ContentGenerator interface
 * This implementation can be used as a base for other OpenAI-compatible providers
 */
export class OpenAIProvider extends AbstractProvider {
  readonly provider = LLMProvider.OPENAI;
  protected baseURL: string;
  protected apiKey: string;

  constructor(config: ProviderConfig) {
    super(config);
    this.baseURL = config.baseURL || 'https://api.openai.com/v1';
    this.apiKey = config.apiKey || '';
  }

  async initialize(): Promise<void> {
    if (!this.apiKey) {
      throw new ProviderConfigurationError(
        this.provider,
        'OpenAI API key is required. Please set apiKey in provider config.',
      );
    }

    // Test API key by making a simple request
    try {
      await this.validateConfig();
    } catch (error) {
      throw new ProviderConfigurationError(
        this.provider,
        `Failed to initialize OpenAI provider: ${error instanceof Error ? error.message : 'Unknown error'}`,
      );
    }
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    try {
      const messages = this.convertGeminiToOpenAIMessages(request.contents);
      const model = request.model || this.config.model || 'gpt-4o';
      const tools = this.convertGeminiToolsToOpenAITools(request.config?.tools);

      const openaiRequest: OpenAIChatRequest = {
        model,
        messages,
        temperature: request.config?.temperature || 0.7,
        max_tokens: request.config?.maxOutputTokens || undefined,
        stream: false,
        stop: request.config?.stopSequences || undefined,
      };

      // Add tools if available
      if (tools && tools.length > 0) {
        openaiRequest.tools = tools;
        openaiRequest.tool_choice = 'auto';
      }

      const response = await this.makeOpenAIRequest(
        '/chat/completions',
        openaiRequest,
        {
          signal: request.config?.abortSignal,
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data: OpenAIChatResponse = await response.json();
      return this.convertOpenAIToGeminiResponse(data);
    } catch (error) {
      throw new ProviderAPIError(
        this.provider,
        `Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.convertGeminiToOpenAIMessages(request.contents);
    const model = request.model || this.config.model || 'gpt-4o';
    const tools = this.convertGeminiToolsToOpenAITools(request.config?.tools);

    const openaiRequest: OpenAIChatRequest = {
      model,
      messages,
      temperature: request.config?.temperature || 0.7,
      max_tokens: request.config?.maxOutputTokens || undefined,
      stream: true,
      stop: request.config?.stopSequences || undefined,
    };

    // Add tools if available
    if (tools && tools.length > 0) {
      openaiRequest.tools = tools;
      openaiRequest.tool_choice = 'auto';
    }

    try {
      const response = await this.makeOpenAIRequest(
        '/chat/completions',
        openaiRequest,
        {
          signal: request.config?.abortSignal,
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      return this.parseOpenAIStream(response);
    } catch (error) {
      throw new ProviderAPIError(
        this.provider,
        `Streaming failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // OpenAI doesn't have a direct token counting API
    // Estimate using character count (rough approximation: ~4 characters per token)
    const text = this.extractTextFromContents(request.contents);
    const estimatedTokens = Math.ceil(text.length / 4);

    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    try {
      const texts = this.convertEmbedContentToTexts(
        Array.isArray(request.contents) ? request.contents : [request.contents],
      );
      const model = 'text-embedding-3-small'; // Default embedding model

      const embeddingRequest: OpenAIEmbeddingRequest = {
        model,
        input: texts,
      };

      const response = await this.makeOpenAIRequest(
        '/embeddings',
        embeddingRequest,
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data: OpenAIEmbeddingResponse = await response.json();

      return {
        embeddings: data.data.map((item) => ({
          values: item.embedding,
        })),
      };
    } catch (error) {
      throw new ProviderAPIError(
        this.provider,
        `Embedding failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        error instanceof Error ? error : undefined,
      );
    }
  }

  supportsEmbeddings(): boolean {
    return true;
  }

  async validateConfig(): Promise<boolean> {
    try {
      // Test with a simple models list request
      const response = await this.makeOpenAIRequest('/models', null, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  protected requiresApiKey(): boolean {
    return true;
  }

  // Protected methods that can be overridden by subclasses (like Ollama)

  protected async makeOpenAIRequest(
    endpoint: string,
    body: unknown,
    options: { method?: string; signal?: AbortSignal } = {},
  ): Promise<Response> {
    const { method = 'POST', signal } = options;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Add authorization header if API key is provided
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

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

  protected convertGeminiToOpenAIMessages(
    contents: ContentListUnion,
  ): OpenAIMessage[] {
    const contentArray = Array.isArray(contents) ? contents : [contents];
    const messages: OpenAIMessage[] = [];

    for (const content of contentArray) {
      if (typeof content === 'string') {
        messages.push({
          role: 'user' as const,
          content,
        });
        continue;
      }

      if ('role' in content && 'parts' in content && content.parts) {
        const role =
          content.role === 'model'
            ? 'assistant'
            : (content.role as 'user' | 'system');

        // Check if this is a function call message
        const toolCalls = this.extractToolCallsFromParts(content.parts);
        if (toolCalls.length > 0) {
          messages.push({
            role: 'assistant',
            content: null,
            tool_calls: toolCalls,
          });
          continue;
        }

        // Check if this is a function response message
        const functionResponseParts = content.parts.filter(
          (part) =>
            part && typeof part === 'object' && 'functionResponse' in part,
        );

        if (functionResponseParts.length > 0) {
          // Each function response becomes a separate tool message
          for (const part of functionResponseParts) {
            if (
              part &&
              typeof part === 'object' &&
              'functionResponse' in part
            ) {
              const functionResponse = part.functionResponse;
              messages.push({
                role: 'tool',
                content: JSON.stringify(functionResponse?.response || {}),
                tool_call_id: functionResponse?.id || `unknown-${Date.now()}`,
              });
            }
          }
          continue;
        }

        // Regular text message
        messages.push({
          role,
          content: this.extractTextFromContent(content),
        });
      } else {
        // Handle Part objects directly
        messages.push({
          role: 'user' as const,
          content: this.extractTextFromContent(content as Content),
        });
      }
    }

    return messages;
  }

  protected convertOpenAIToGeminiResponse(
    response: OpenAIChatResponse,
  ): GenerateContentResponse {
    const choice = response.choices?.[0];
    if (!choice) {
      throw new Error('Invalid response format from OpenAI');
    }

    const parts: Part[] = [];

    // Handle text content
    if (choice.message?.content) {
      parts.push({ text: choice.message.content });
    }

    // Handle tool calls
    if (choice.message?.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        if (toolCall.function) {
          parts.push({
            functionCall: {
              id: toolCall.id,
              name: toolCall.function.name,
              args: this.parseToolCallArguments(toolCall.function.arguments),
            },
          } as Part);
        }
      }
    }

    // If no parts were added, add an empty text part
    if (parts.length === 0) {
      parts.push({ text: '' });
    }

    const out = new GenerateContentResponse();
    out.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason: this.convertFinishReason(choice.finish_reason),
        index: 0,
      },
    ];
    out.usageMetadata = {
      promptTokenCount: response.usage?.prompt_tokens || 0,
      candidatesTokenCount: response.usage?.completion_tokens || 0,
      totalTokenCount: response.usage?.total_tokens || 0,
    };
    return out;
  }

  protected async *parseOpenAIStream(
    response: Response,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body available for streaming');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let currentToolCall: {
      id: string;
      name: string;
      arguments: string;
    } | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('data: ')) {
            const jsonData = trimmedLine.slice(6); // Remove "data: " prefix
            if (jsonData === '[DONE]') {
              // Complete any remaining tool call at the end of the stream
              if (
                currentToolCall &&
                currentToolCall.id &&
                currentToolCall.name
              ) {
                const response = new GenerateContentResponse();
                response.candidates = [
                  {
                    content: {
                      parts: [
                        {
                          functionCall: {
                            id: currentToolCall.id,
                            name: currentToolCall.name,
                            args: this.parseToolCallArguments(
                              currentToolCall.arguments,
                            ),
                          },
                        } as Part,
                      ],
                      role: 'model',
                    },
                    finishReason: FinishReason.STOP,
                    index: 0,
                  },
                ];
                yield response;
              }
              return;
            }

            try {
              const chunk: OpenAIStreamChunk = JSON.parse(jsonData);
              const choice = chunk.choices?.[0];
              if (!choice) continue;

              const parts: Part[] = [];

              // Handle text content
              if (choice.delta.content) {
                parts.push({ text: choice.delta.content });
              }

              // Handle tool calls (streaming)
              if (choice.delta.tool_calls) {
                for (const toolCall of choice.delta.tool_calls) {
                  if (toolCall.function) {
                    // Start of a new tool call
                    if (toolCall.function.name) {
                      // Complete previous tool call if it exists
                      if (
                        currentToolCall &&
                        currentToolCall.id &&
                        currentToolCall.name
                      ) {
                        parts.push({
                          functionCall: {
                            id: currentToolCall.id,
                            name: currentToolCall.name,
                            args: this.parseToolCallArguments(
                              currentToolCall.arguments,
                            ),
                          },
                        } as Part);
                      }

                      // Start new tool call
                      currentToolCall = {
                        id: toolCall.id,
                        name: toolCall.function.name,
                        arguments: toolCall.function.arguments || '',
                      };
                    }
                    // Continuation of arguments
                    else if (currentToolCall && toolCall.function.arguments) {
                      currentToolCall.arguments += toolCall.function.arguments;
                    }
                  }
                }
              }

              // Complete tool call when we have finish_reason or no more content
              if (
                currentToolCall &&
                (choice.finish_reason ||
                  (!choice.delta.tool_calls && !choice.delta.content))
              ) {
                parts.push({
                  functionCall: {
                    id: currentToolCall.id,
                    name: currentToolCall.name,
                    args: this.parseToolCallArguments(
                      currentToolCall.arguments,
                    ),
                  },
                } as Part);
                currentToolCall = null;
              }

              if (parts.length > 0) {
                const streamOut = new GenerateContentResponse();
                streamOut.candidates = [
                  {
                    content: {
                      parts,
                      role: 'model',
                    },
                    finishReason: choice.finish_reason
                      ? this.convertFinishReason(choice.finish_reason)
                      : FinishReason.STOP,
                    index: choice.index,
                  },
                ];

                if (chunk.usage) {
                  streamOut.usageMetadata = {
                    promptTokenCount: chunk.usage.prompt_tokens,
                    candidatesTokenCount: chunk.usage.completion_tokens,
                    totalTokenCount: chunk.usage.total_tokens,
                  };
                }

                yield streamOut;
              }
            } catch {
              // Skip invalid JSON lines
              console.warn(
                'Failed to parse streaming response line:',
                jsonData,
              );
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  protected convertFinishReason(reason: string): FinishReason | undefined {
    switch (reason) {
      case 'stop':
        return FinishReason.STOP;
      case 'length':
        return FinishReason.MAX_TOKENS;
      case 'content_filter':
        return FinishReason.SAFETY;
      case 'tool_calls':
        return FinishReason.STOP; // OpenAI function calls
      default:
        return undefined;
    }
  }

  protected extractTextFromContent(content: Content | string): string {
    if (typeof content === 'string') {
      return content;
    }

    if ('parts' in content && content.parts) {
      return content.parts
        .map((part) => (typeof part === 'string' ? part : part.text))
        .filter(Boolean)
        .join('');
    }

    return '';
  }

  protected extractTextFromContents(contents: ContentListUnion): string {
    const contentArray = Array.isArray(contents) ? contents : [contents];
    return contentArray
      .map((content) =>
        typeof content === 'string'
          ? content
          : this.extractTextFromContent(content as Content),
      )
      .join(' ');
  }

  protected convertEmbedContentToTexts(contents: unknown[]): string[] {
    return contents.map((content) =>
      typeof content === 'string'
        ? content
        : this.extractTextFromContent(content as Content),
    );
  }

  protected convertGeminiToolsToOpenAITools(
    tools: ToolListUnion | undefined,
  ): OpenAITool[] | undefined {
    if (!tools || tools.length === 0) {
      return undefined;
    }

    // Extract FunctionDeclarations from ToolListUnion
    const functionDeclarations: FunctionDeclaration[] = [];
    for (const tool of tools) {
      if ('functionDeclarations' in tool && tool.functionDeclarations) {
        functionDeclarations.push(...tool.functionDeclarations);
      }
    }

    if (functionDeclarations.length === 0) {
      return undefined;
    }

    return functionDeclarations.map((tool) => ({
      type: 'function' as const,
      function: {
        name: tool.name || '',
        description: tool.description || '',
        parameters: (tool.parameters as Record<string, unknown>) || {},
      },
    }));
  }

  protected extractToolCallsFromParts(parts: Part[]): OpenAIToolCall[] {
    const toolCalls: OpenAIToolCall[] = [];

    for (const part of parts) {
      if (part.functionCall && part.functionCall.name) {
        toolCalls.push({
          id:
            part.functionCall.id ||
            `call_${Date.now()}_${Math.random().toString(16).slice(2)}`,
          type: 'function',
          function: {
            name: part.functionCall.name,
            arguments: JSON.stringify(part.functionCall.args || {}),
          },
        });
      }
    }

    return toolCalls;
  }

  protected parseToolCallArguments(
    argumentsStr: string,
  ): Record<string, unknown> {
    if (!argumentsStr || argumentsStr.trim() === '') {
      return {};
    }

    try {
      return JSON.parse(argumentsStr);
    } catch (error) {
      console.warn(
        '[OpenAI] Failed to parse tool call arguments:',
        argumentsStr,
        error,
      );
      return {};
    }
  }
}
